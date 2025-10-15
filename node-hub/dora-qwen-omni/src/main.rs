//! Based on the mtmd cli example from llama.cpp.

use base64::Engine;
use clap::Parser;
use eyre::{Context, Result};
use image::DynamicImage;
use img_hash::{HashBytes, HasherConfig, ImageHash};
use std::ffi::CString;
use std::io::Cursor;
use std::num::NonZeroU32;
use std::path::Path;

use dora_node_api::arrow::array::AsArray;
use dora_node_api::dora_core::config::DataId;
use dora_node_api::{DoraNode, Event, IntoArrow};
use image::{io::Reader, Rgba};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::mtmd::{
    MtmdBitmap, MtmdBitmapError, MtmdContext, MtmdContextParams, MtmdInputText,
};

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::{LlamaChatMessage, LlamaChatTemplate, LlamaModel, Special};
use llama_cpp_2::sampling::LlamaSampler;
use llm_json::{repair_json, RepairOptions};
use rusttype::{Font, Scale};
use serde::{Deserialize, Serialize};
use text_on_image::{text_on_image_with_background, FontBundle};

/// Command line parameters for the MTMD CLI application
#[derive(clap::Parser, Debug)]
#[command(name = "mtmd-cli")]
#[command(about = "Experimental CLI for multimodal llama.cpp")]
pub struct MtmdCliParams {
    /// Path to the model file
    #[arg(short = 'm', long = "model", value_name = "PATH")]
    pub model_path: String,
    /// Path to the multimodal projection file
    #[arg(long = "mmproj", value_name = "PATH")]
    pub mmproj_path: String,
    /// Path to image file(s)
    #[arg(long = "image", value_name = "PATH")]
    pub images: Vec<String>,
    /// Path to audio file(s)
    #[arg(long = "audio", value_name = "PATH")]
    pub audio: Vec<String>,
    /// Text prompt to use as input to the model. May include media markers - else they will be added automatically.
    #[arg(short = 'p', long = "prompt", value_name = "TEXT")]
    pub prompt: String,
    /// Number of tokens to predict (-1 for unlimited)
    #[arg(
        short = 'n',
        long = "n-predict",
        value_name = "N",
        default_value = "4096"
    )]
    pub n_predict: i32,
    /// Number of threads
    #[arg(short = 't', long = "threads", value_name = "N", default_value = "128")]
    pub n_threads: i32,
    /// Maximum number of tokens in context
    #[arg(long = "n-tokens", value_name = "N", default_value = "16384")]
    pub n_tokens: NonZeroU32,
    /// Chat template to use, default template if not provided
    #[arg(long = "chat-template", value_name = "TEMPLATE")]
    pub chat_template: Option<String>,
    /// Disable GPU acceleration
    #[arg(long = "no-gpu")]
    pub no_gpu: bool,
    /// Disable GPU offload for multimodal projection
    #[arg(long = "no-mmproj-offload")]
    pub no_mmproj_offload: bool,
    /// Media marker. If not provided, the default marker will be used.
    #[arg(long = "marker", value_name = "TEXT")]
    pub media_marker: Option<String>,
}

/// State of the MTMD CLI application.
#[allow(missing_debug_implementations)]
pub struct MtmdCliContext {
    /// The MTMD context for multimodal processing.
    pub mtmd_ctx: MtmdContext,
    /// The batch used for processing tokens.
    pub batch: LlamaBatch,
    /// The list of loaded bitmaps (images/audio).
    pub bitmaps: Vec<MtmdBitmap>,
    /// The number of past tokens processed.
    pub n_past: i32,
    /// The chat template used for formatting messages.
    pub chat_template: LlamaChatTemplate,
    /// The current chat messages history.
    pub chat: Vec<LlamaChatMessage>,
}

impl MtmdCliContext {
    /// Creates a new MTMD CLI context
    ///
    /// # Errors
    pub fn new(
        params: &MtmdCliParams,
        model: &LlamaModel,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Initialize MTMD context
        let mtmd_params = MtmdContextParams {
            use_gpu: !params.no_gpu && !params.no_mmproj_offload,
            print_timings: true,
            n_threads: params.n_threads,
            media_marker: CString::new(
                params
                    .media_marker
                    .as_ref()
                    .unwrap_or(&llama_cpp_2::mtmd::mtmd_default_marker().to_string())
                    .clone(),
            )?,
        };

        let mtmd_ctx = MtmdContext::init_from_file(&params.mmproj_path, model, &mtmd_params)?;

        let chat_template = model
            .chat_template(params.chat_template.as_deref())
            .map_err(|e| format!("Failed to get chat template: {e}"))?;

        let batch = LlamaBatch::new(params.n_tokens.get() as usize, 1);

        Ok(Self {
            mtmd_ctx,
            batch,
            chat: Vec::new(),
            bitmaps: Vec::new(),
            n_past: 0,
            chat_template,
        })
    }

    /// Loads media (image or audio) from the specified file path
    /// # Errors
    pub fn load_media(&mut self, path: &str) -> Result<()> {
        let bitmap = MtmdBitmap::from_file(&self.mtmd_ctx, path)
            .context("failed to load image from path")?;
        self.bitmaps.push(bitmap);
        Ok(())
    }

    /// Loads media (image or audio) from the specified file path
    /// # Errors
    pub fn load_base64_image(&mut self, buffer: &[u8]) -> Result<()> {
        let bitmap =
            MtmdBitmap::from_buffer(&self.mtmd_ctx, buffer).context("Failed to load image")?;
        self.bitmaps.push(bitmap);
        Ok(())
    }

    pub fn clear_chat(&mut self) -> Result<(), MtmdBitmapError> {
        self.chat.clear();
        self.bitmaps.clear();
        self.n_past = 0;
        Ok(())
    }

    /// Evaluates a chat message, tokenizing and processing it through the model
    /// # Errors
    pub fn eval_message(
        &mut self,
        model: &LlamaModel,
        context: &mut LlamaContext,
        msg: LlamaChatMessage,
        add_bos: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.chat.push(msg);

        // Format the message using chat template (simplified)
        let formatted_prompt = model.apply_chat_template(&self.chat_template, &self.chat, true)?;

        let input_text = MtmdInputText {
            text: formatted_prompt,
            add_special: add_bos,
            parse_special: true,
        };

        let bitmap_refs: Vec<&MtmdBitmap> = self.bitmaps.iter().collect();

        let batch = if bitmap_refs.is_empty() {
            println!("No bitmaps provided, only tokenizing text");
            1
        } else {
            println!("Tokenizing with {} bitmaps", bitmap_refs.len());
            1
        };

        // Tokenize the input
        let chunks = self.mtmd_ctx.tokenize(input_text, &bitmap_refs)?;

        // Clear bitmaps after tokenization
        self.bitmaps.clear();

        self.n_past = chunks.eval_chunks(&self.mtmd_ctx, context, self.n_past, 0, batch, true)?;
        Ok(())
    }

    /// Generates a response by sampling tokens from the model
    /// # Errors
    pub fn generate_response(
        &mut self,
        model: &LlamaModel,
        context: &mut LlamaContext,
        sampler: &mut LlamaSampler,
        n_predict: i32,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let mut generated_tokens = Vec::new();
        let max_predict = if n_predict < 0 { i32::MAX } else { n_predict };
        let mut text = String::new();
        for _i in 0..max_predict {
            // Sample next token
            let token = sampler.sample(context, 0);

            generated_tokens.push(token);

            sampler.accept(token);

            // Check for end of generation
            if model.is_eog_token(token) {
                break;
            }

            // Print token
            let piece = model
                .token_to_str(token, Special::Plaintext)
                .unwrap_or_default();
            text.push_str(&piece);

            // Prepare next batch
            self.batch.clear();
            self.batch.add(token, self.n_past, &[0], true)?;
            self.n_past += 1;

            // Decode
            context.decode(&mut self.batch)?;
        }

        Ok(text)
    }
}

fn run_single_turn(
    model: &LlamaModel,
    context: &mut LlamaContext,
    sampler: &mut LlamaSampler,
    params: &MtmdCliParams,
    backend: &LlamaBackend,
) -> Result<(), Box<dyn std::error::Error>> {
    // Add media marker if not present
    let (mut node, mut events) = DoraNode::init_from_env().unwrap();
    let mut ctx = MtmdCliContext::new(&params, &model)?;
    let mut never_debounced = true;
    loop {
        match events.recv() {
            Some(Event::Input {
                id: _,
                data,
                metadata: _,
            }) => {
                ctx.clear_chat().unwrap();
                context.clear_kv_cache();
                let instant = std::time::Instant::now();
                let default_marker = llama_cpp_2::mtmd::mtmd_default_marker().to_string();
                let texts = data.as_string::<i32>();
                let mut prompt = "".to_string();
                let mut img = None;
                for text in texts {
                    match text {
                        Some(text) => {
                            if text.starts_with("<|user|>\n<|im_start|>\n") {
                                prompt.push_str(&text.replace("<|user|>\n<|im_start|>\n", ""));
                            } else if text.starts_with("<|user|>\n<|vision_start|>\n") {
                                let string = text.replace("<|user|>\n<|vision_start|>\n", "");
                                println!("got image string: {}", string);
                                if string.starts_with("data:image/") {
                                    let mut string = string.split(",");
                                    let _encoding = string.next().unwrap();
                                    let data = string.next().unwrap();
                                    let engine = base64::engine::general_purpose::STANDARD;
                                    let decoded_data = engine.decode(data)?;
                                    img = Some(
                                        Reader::new(Cursor::new(decoded_data.clone()))
                                            .with_guessed_format()?
                                            .decode()?,
                                    );
                                    ctx.load_base64_image(&decoded_data)?;
                                    prompt.push_str(&default_marker);
                                    println!("got base64 image----------------------------");
                                } else if url::Url::parse(&string).is_ok() {
                                    let resp = reqwest::blocking::get(&string);
                                    match resp {
                                        Ok(resp) => {
                                            let bytes = resp.bytes().unwrap();
                                            img = Some(
                                                Reader::new(Cursor::new(bytes.clone()))
                                                    .with_guessed_format()?
                                                    .decode()?,
                                            );
                                            let engine = base64::engine::general_purpose::STANDARD;
                                            ctx.load_base64_image(&bytes)?;
                                            prompt.push_str(&default_marker);
                                        }
                                        Err(e) => {
                                            println!("failed to fetch image from url: {}", e);
                                        }
                                    }
                                }
                            } else if text.starts_with("<|system|>\n") {
                                continue;
                            } else {
                                prompt.push_str(&text.replace("<|user|>\n<|im_start|>\n", ""));
                            }
                        }
                        None => {}
                    }
                }
                // Create user message
                let msg = LlamaChatMessage::new("user".to_string(), prompt)?;

                // Evaluate the message (prefill)
                let instant2 = std::time::Instant::now();
                ctx.eval_message(model, context, msg, true)?;
                let text = ctx.generate_response(model, context, sampler, params.n_predict)?;
                let elapsed = instant2.elapsed();
                println!("got token in {:.2?}", elapsed);
                ctx.clear_chat().unwrap();
                context.clear_kv_cache();

                node.send_output(
                    DataId::from("text".to_string()),
                    Default::default(),
                    text.clone().into_arrow(),
                )?;
            }
            _ => break,
        }
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let params = MtmdCliParams::parse();

    // Validate required parameters
    if !Path::new(&params.model_path).exists() {
        eprintln!("Error: Model file not found: {}", params.model_path);
        return Err("Model file not found".into());
    }

    if !Path::new(&params.mmproj_path).exists() {
        eprintln!(
            "Error: Multimodal projection file not found: {}",
            params.mmproj_path
        );
        return Err("Multimodal projection file not found".into());
    }

    // Initialize backend
    let backend = LlamaBackend::init()?;

    // Setup model parameters
    let mut model_params = LlamaModelParams::default();
    if !params.no_gpu {
        model_params = model_params.with_n_gpu_layers(1_000_000); // Use all layers on GPU
    }

    // Load model
    let model = LlamaModel::load_from_file(&backend, &params.model_path, &model_params)?;

    // Create context
    let context_params = LlamaContextParams::default()
        .with_n_threads(params.n_threads)
        .with_n_batch(64)
        .with_n_ctx(Some(params.n_tokens));

    let mut context = model.new_context(&backend, context_params)?;
    // Create sampler
    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);

    // Create the MTMD context
    run_single_turn(&model, &mut context, &mut sampler, &params, &backend)?;

    Ok(())
}
