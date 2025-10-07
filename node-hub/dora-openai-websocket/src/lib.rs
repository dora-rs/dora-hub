use base64::Engine;
use base64::engine::general_purpose;
use dora_node_api::ArrowData;
use dora_node_api::DoraNode;
use dora_node_api::Event;
use dora_node_api::IntoArrow;
use dora_node_api::MetadataParameters;
use dora_node_api::arrow::array::Array;
use dora_node_api::arrow::array::ArrayData;
use dora_node_api::arrow::array::ArrayRef;
use dora_node_api::arrow::array::AsArray;
use dora_node_api::arrow::array::make_array;
use dora_node_api::arrow::datatypes::DataType;
use dora_node_api::dora_core::config::DataId;
use dora_node_api::into_vec;
use fastwebsockets::Frame;
use fastwebsockets::OpCode;
use fastwebsockets::Payload;
use fastwebsockets::WebSocketError;
use fastwebsockets::upgrade;
use futures_concurrency::future::Race;
use futures_util::FutureExt;
use futures_util::future;
use futures_util::future::Either;
use http_body_util::Empty;
use hyper::Request;
use hyper::Response;
use hyper::body::Bytes;
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use serde;
use serde::Deserialize;
use serde::Serialize;
use serde_json::value::RawValue;
use tokio::net::TcpListener;

#[derive(Serialize, Deserialize, Debug)]
pub struct ErrorDetails {
    pub code: Option<String>,
    pub message: String,
    pub param: Option<String>,
    #[serde(rename = "type")]
    pub error_type: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
pub enum OpenAIRealtimeMessage {
    #[serde(rename = "session.update")]
    SessionUpdate { session: SessionConfig },
    #[serde(rename = "input_audio_buffer.append")]
    InputAudioBufferAppend {
        audio: String, // base64 encoded audio
    },
    #[serde(rename = "input_audio_buffer.commit")]
    InputAudioBufferCommit,
    #[serde(rename = "response.create")]
    ResponseCreate {
        #[serde(default)]
        response: ResponseConfig,
    },
    #[serde(rename = "conversation.item.create")]
    ConversationItemCreate { item: ConversationItem },
    #[serde(rename = "conversation.item.truncate")]
    ConversationItemTruncate {
        item_id: String,
        content_index: u32,
        audio_end_ms: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
    },
}

fn default_model() -> String {
    "Qwen/Qwen2.5-3B-Instruct-GGUF".to_string()
}
#[derive(Serialize, Deserialize, Debug)]
pub struct SessionConfig {
    #[serde(default)]
    pub modalities: Vec<String>,
    #[serde(default)]
    pub instructions: String,
    #[serde(default)]
    pub voice: String,
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default)]
    pub input_audio_format: String,
    #[serde(default)]
    pub output_audio_format: String,
    #[serde(default)]
    pub input_audio_transcription: Option<TranscriptionConfig>,
    #[serde(default)]
    pub turn_detection: Option<TurnDetectionConfig>,
    #[serde(default)]
    pub tools: Vec<serde_json::Value>,
    #[serde(default)]
    pub tool_choice: String,
    #[serde(default)]
    pub temperature: f32,
    #[serde(default)]
    pub max_response_output_tokens: Option<u32>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TranscriptionConfig {
    pub model: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TurnDetectionConfig {
    #[serde(default)]
    #[serde(rename = "type")]
    pub detection_type: String,
    #[serde(default)]
    pub threshold: f32,
    #[serde(default)]
    pub prefix_padding_ms: u32,
    #[serde(default)]
    pub silence_duration_ms: u32,
    #[serde(default)]
    pub interrupt_response: bool,
    #[serde(default)]
    pub create_response: bool,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct ResponseConfig {
    #[serde(default)]
    pub modalities: Vec<String>,
    pub instructions: Option<String>,
    pub voice: Option<String>,
    pub output_audio_format: Option<String>,
    pub tools: Option<serde_json::Value>,
    pub tool_choice: Option<String>,
    pub temperature: Option<f32>,
    pub max_output_tokens: Option<u32>,
}

#[derive(Deserialize, Serialize, Debug)]
#[serde(tag = "type")]
pub enum ResponseOutputItem {
    #[serde(rename = "function_call")]
    FunctionCall {
        id: String,
        name: String,
        call_id: String,
        arguments: String,
        status: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct ResponseDoneData {
    pub id: String,
    pub status: String,
    pub output: Vec<ResponseOutputItem>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ConversationItem {
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub item_type: String, // "message", "function_call", "function_call_output"
    pub object: Option<String>,
    pub status: Option<String>, // "completed", "in_progress", "incomplete"
    pub role: Option<String>,   // "user", "assistant", "system"
    #[serde(default)]
    pub content: Vec<ContentPart>,
    pub call_id: Option<String>,
    pub output: Option<String>,
    pub name: Option<String>,
    pub arguments: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "input_text")]
    InputText { text: String },
    #[serde(rename = "input_audio")]
    InputAudio {
        audio: String,
        transcript: Option<String>,
    },
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "audio")]
    Audio {
        audio: String,
        transcript: Option<String>,
    },
}

// Implement simple tool definition
#[derive(Serialize, Deserialize, Debug)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Box<RawValue>, // Owned RawValue
}

// Incoming message types from OpenAI
#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
pub enum OpenAIRealtimeResponse {
    #[serde(rename = "error")]
    Error { error: ErrorDetails },
    #[serde(rename = "session.created")]
    SessionCreated { session: serde_json::Value },
    #[serde(rename = "session.updated")]
    SessionUpdated { session: serde_json::Value },
    #[serde(rename = "conversation.item.created")]
    ConversationItemCreated { item: serde_json::Value },
    #[serde(rename = "conversation.item.truncated")]
    ConversationItemTruncated { item: serde_json::Value },
    #[serde(rename = "response.audio.delta")]
    ResponseAudioDelta {
        response_id: String,
        item_id: String,
        output_index: u32,
        content_index: u32,
        delta: String, // base64 encoded audio
    },
    #[serde(rename = "response.audio.done")]
    ResponseAudioDone {
        response_id: String,
        item_id: String,
        output_index: u32,
        content_index: u32,
    },
    #[serde(rename = "response.function_call_arguments.done")]
    ResponseFunctionCallArgumentsDone {
        item_id: String,
        output_index: u32,
        sequence_number: u32,
        call_id: String,
        name: String,
        arguments: String,
    },
    #[serde(rename = "response.function_call_arguments.delta")]
    ResponseFunctionCallArgumentsDelta {
        response_id: String,
        item_id: String,
        output_index: u32,
        call_id: String,
        delta: String,
    },
    #[serde(rename = "response.output_item.added")]
    ResponseOutputItemAdded {
        event_id: String,
        response_id: String,
        output_index: u32,
        item: ConversationItem,
    },
    #[serde(rename = "response.text.delta")]
    ResponseTextDelta {
        response_id: String,
        item_id: String,
        output_index: u32,
        content_index: u32,
        delta: String,
    },
    #[serde(rename = "response.audio_transcript.delta")]
    ResponseAudioTranscriptDelta {
        response_id: String,
        item_id: String,
        output_index: u32,
        content_index: u32,
        delta: String,
    },
    #[serde(rename = "response.done")]
    ResponseDone { response: ResponseDoneData },
    #[serde(rename = "input_audio_buffer.speech_started")]
    InputAudioBufferSpeechStarted {
        audio_start_ms: u32,
        item_id: String,
    },
    #[serde(rename = "input_audio_buffer.speech_stopped")]
    InputAudioBufferSpeechStopped { audio_end_ms: u32, item_id: String },
    #[serde(other)]
    Other,
}

fn convert_pcm16_to_f32(bytes: &[u8]) -> Vec<f32> {
    let mut samples = Vec::with_capacity(bytes.len() / 2);

    for chunk in bytes.chunks_exact(2) {
        let pcm16_sample = i16::from_le_bytes([chunk[0], chunk[1]]);
        let f32_sample = pcm16_sample as f32 / 32767.0;
        samples.push(f32_sample);
    }

    samples
}

fn convert_f32_to_pcm16(samples: &[f32]) -> Vec<u8> {
    let mut pcm16_bytes = Vec::with_capacity(samples.len() * 2);

    for &sample in samples {
        // Clamp to [-1.0, 1.0] and convert to i16
        let clamped = sample.max(-1.0).min(1.0);
        let pcm16_sample = (clamped * 32767.0) as i16;
        pcm16_bytes.extend_from_slice(&pcm16_sample.to_le_bytes());
    }

    pcm16_bytes
}

#[derive(Debug, Clone)]
enum DoraTokioBroadcast {
    Output(DataId, MetadataParameters, ArrayData),
    Input(DataId, MetadataParameters, ArrayRef),
}

async fn handle_client(
    fut: upgrade::UpgradeFut,
    tx: tokio::sync::broadcast::Sender<DoraTokioBroadcast>,
) -> Result<(), WebSocketError> {
    let mut ws = fastwebsockets::FragmentCollector::new(fut.await?);

    let frame = ws.read_frame().await?;
    if frame.opcode != OpCode::Text {
        return Err(WebSocketError::InvalidConnectionHeader);
    }
    let data: OpenAIRealtimeMessage = serde_json::from_slice(&frame.payload).unwrap();
    let OpenAIRealtimeMessage::SessionUpdate { session } = data else {
        return Err(WebSocketError::InvalidConnectionHeader);
    };
    let system_prompt = session.instructions.clone();
    let tools = serde_json::to_string(&session.tools.clone()).unwrap_or_default();

    // Copy configuration file but replace the node ID with "server-id"
    // Read the configuration file and replace the node ID with "server-id"
    let serialized_data = OpenAIRealtimeResponse::SessionCreated {
        session: serde_json::Value::Null,
    };

    tx.send(DoraTokioBroadcast::Output(
        DataId::from("system_prompt".to_string()),
        MetadataParameters::default(),
        system_prompt.into_arrow().to_data(),
    ))
    .unwrap();
    tx.send(DoraTokioBroadcast::Output(
        DataId::from("tools".to_string()),
        MetadataParameters::default(),
        tools.into_arrow().to_data(),
    ))
    .unwrap();
    let payload =
        Payload::Bytes(Bytes::from(serde_json::to_string(&serialized_data).unwrap()).into());
    let frame = Frame::text(payload);
    ws.write_frame(frame).await?;

    // Local variable

    let mut call_id = 0;
    let mut item_id = 0;
    loop {
        let mut rx = tx.subscribe();
        let event_fut = rx.recv().map(Either::Left);
        let frame_fut = ws.read_frame().map(Either::Right);
        let event_stream = (event_fut, frame_fut).race();
        let frame = match event_stream.await {
            future::Either::Left(Ok(DoraTokioBroadcast::Input(id, _metadata, data))) => {
                let frame = if data.data_type() == &DataType::Utf8 && id.contains("transcript") {
                    let data = data.as_string::<i32>();
                    let str = data.value(0);
                    let serialized_data = OpenAIRealtimeResponse::ResponseAudioTranscriptDelta {
                        response_id: "123".to_string(),
                        item_id: item_id.to_string(),
                        output_index: 123,
                        content_index: 123,
                        delta: str.to_string(),
                    };
                    item_id += 1;

                    let frame = Frame::text(Payload::Bytes(
                        Bytes::from(serde_json::to_string(&serialized_data).unwrap()).into(),
                    ));
                    frame
                } else if data.data_type() == &DataType::Utf8 && id.contains("text") {
                    let data = data.as_string::<i32>();
                    let orig_str = data.value(0);
                    // If response start and finish with <tool_call> parse it.
                    let frame = if orig_str.starts_with("<tool_call>") {
                        let str = orig_str
                            .trim_start_matches("<tool_call>")
                            .trim_end_matches("</tool_call>");

                        // Replace double curly braces with single curly braces
                        let str = if str.contains("{{") {
                            str.replace("{{", "{").replace("}}}", "}}")
                        } else {
                            str.to_string()
                        };

                        if let Ok(tool_call) = serde_json::from_str::<ToolCall>(&str) {
                            let serialized_data = OpenAIRealtimeResponse::ResponseOutputItemAdded {
                                event_id: "123".to_string(),
                                response_id: "123".to_string(),
                                output_index: 123,
                                item: ConversationItem {
                                    id: Some("msg_007".to_string()),
                                    item_type: "function_call".to_string(),
                                    status: Some("in_progress".to_string()),
                                    role: Some("assistant".to_string()),
                                    content: vec![],
                                    call_id: call_id.to_string().into(),
                                    output: None,
                                    name: Some(tool_call.name.clone()),
                                    arguments: None,
                                    object: None,
                                },
                            };
                            let frame = Frame::text(Payload::Bytes(
                                Bytes::from(serde_json::to_string(&serialized_data).unwrap())
                                    .into(),
                            ));

                            ws.write_frame(frame).await.unwrap();
                            let serialized_data =
                                OpenAIRealtimeResponse::ResponseFunctionCallArgumentsDelta {
                                    item_id: item_id.to_string(),
                                    output_index: 123,
                                    call_id: call_id.to_string().into(),
                                    response_id: "123".to_string(),
                                    delta: tool_call.arguments.to_string(),
                                };
                            item_id += 1;
                            let frame = Frame::text(Payload::Bytes(
                                Bytes::from(serde_json::to_string(&serialized_data).unwrap())
                                    .into(),
                            ));

                            ws.write_frame(frame).await.unwrap();

                            let serialized_data =
                                OpenAIRealtimeResponse::ResponseFunctionCallArgumentsDone {
                                    item_id: item_id.to_string(),
                                    output_index: 123,
                                    call_id: call_id.to_string().into(),
                                    sequence_number: 123,
                                    name: tool_call.name,
                                    arguments: tool_call.arguments.to_string(),
                                };
                            call_id += 1;
                            item_id += 1;
                            let frame = Frame::text(Payload::Bytes(
                                Bytes::from(serde_json::to_string(&serialized_data).unwrap())
                                    .into(),
                            ));
                            frame
                        } else {
                            if let Ok(tool_call) = serde_json::from_str::<ToolCall>(&orig_str) {
                                let serialized_data =
                                    OpenAIRealtimeResponse::ResponseFunctionCallArgumentsDone {
                                        item_id: item_id.to_string(),
                                        output_index: 123,
                                        call_id: "123".to_string(),
                                        sequence_number: 123,
                                        name: tool_call.name,
                                        arguments: tool_call.arguments.to_string(),
                                    };
                                item_id += 1;
                                let frame = Frame::text(Payload::Bytes(
                                    Bytes::from(serde_json::to_string(&serialized_data).unwrap())
                                        .into(),
                                ));
                                println!("Sending tool call: {:?}", serialized_data);
                                frame
                            } else {
                                println!("Failed to parse tool call: {}", str);
                                continue;
                            }
                        }
                    } else {
                        let serialized_data = OpenAIRealtimeResponse::ResponseTextDelta {
                            response_id: "123".to_string(),
                            item_id: item_id.to_string(),
                            output_index: 123,
                            content_index: 123,
                            delta: orig_str.to_string(),
                        };
                        item_id += 1;
                        let frame = Frame::text(Payload::Bytes(
                            Bytes::from(serde_json::to_string(&serialized_data).unwrap()).into(),
                        ));
                        frame
                    };
                    frame
                } else if id.contains("audio") {
                    let data: Vec<f32> = into_vec(&ArrowData(data)).unwrap();
                    let data = convert_f32_to_pcm16(&data);
                    let serialized_data = OpenAIRealtimeResponse::ResponseAudioDelta {
                        response_id: "123".to_string(),
                        item_id: item_id.to_string(),
                        output_index: 123,
                        content_index: 123,
                        delta: general_purpose::STANDARD.encode(data),
                    };
                    item_id += 1;
                    let frame = Frame::text(Payload::Bytes(
                        Bytes::from(serde_json::to_string(&serialized_data).unwrap()).into(),
                    ));
                    ws.write_frame(frame).await?;
                    let serialized_data = OpenAIRealtimeResponse::ResponseDone {
                        response: ResponseDoneData {
                            id: "123".to_string(),
                            status: "123".to_string(),
                            output: vec![],
                        },
                    };

                    let payload = Payload::Bytes(
                        Bytes::from(serde_json::to_string(&serialized_data).unwrap()).into(),
                    );
                    println!("Sending response done: {:?}", serialized_data);
                    let frame = Frame::text(payload);
                    frame
                } else if id.contains("speech_started") {
                    let serialized_data = OpenAIRealtimeResponse::InputAudioBufferSpeechStarted {
                        audio_start_ms: 123,
                        item_id: item_id.to_string(),
                    };
                    item_id += 1;

                    let frame = Frame::text(Payload::Bytes(
                        Bytes::from(serde_json::to_string(&serialized_data).unwrap()).into(),
                    ));
                    frame
                } else if id.contains("speech_stopped") {
                    let serialized_data = OpenAIRealtimeResponse::InputAudioBufferSpeechStopped {
                        audio_end_ms: 123,
                        item_id: item_id.to_string(),
                    };
                    item_id += 1;

                    let frame = Frame::text(Payload::Bytes(
                        Bytes::from(serde_json::to_string(&serialized_data).unwrap()).into(),
                    ));
                    frame
                } else {
                    unimplemented!()
                };

                Some(frame)
            }
            future::Either::Left(Ok(DoraTokioBroadcast::Output(_, _, _))) => {
                todo!("Handle Output variant")
            }
            future::Either::Left(Err(_)) => {
                todo!("Handle Error variant")
            }
            future::Either::Right(Ok(frame)) => {
                match frame.opcode {
                    OpCode::Close => break,
                    OpCode::Text | OpCode::Binary => {
                        let data: OpenAIRealtimeMessage =
                            serde_json::from_slice(&frame.payload).unwrap();
                        match data {
                            OpenAIRealtimeMessage::InputAudioBufferAppend { audio } => {
                                // println!("Received audio data: {}", audio);
                                let f32_data = audio;
                                // Decode base64 encoded audio data
                                let f32_data = f32_data.trim();
                                if f32_data.is_empty() {
                                    continue;
                                }

                                if let Ok(f32_data) = general_purpose::STANDARD.decode(f32_data) {
                                    let f32_data = convert_pcm16_to_f32(&f32_data);

                                    let mut parameter = MetadataParameters::default();
                                    parameter.insert(
                                        "sample_rate".to_string(),
                                        dora_node_api::Parameter::Integer(16000),
                                    );
                                    tx.send(DoraTokioBroadcast::Output(
                                        DataId::from("audio".to_string()),
                                        parameter,
                                        f32_data.into_arrow().to_data(),
                                    ))
                                    .unwrap();
                                }
                            }
                            OpenAIRealtimeMessage::InputAudioBufferCommit => break,
                            OpenAIRealtimeMessage::ResponseCreate { response } => {
                                if let Some(text) = response.instructions {
                                    let mut parameter = MetadataParameters::default();
                                    parameter.insert(
                                        "tools".to_string(),
                                        dora_node_api::Parameter::String("[]".to_string()),
                                    );
                                    tx.send(DoraTokioBroadcast::Output(
                                        DataId::from("response.create".to_string()),
                                        parameter,
                                        text.into_arrow().to_data(),
                                    ))
                                    .unwrap();
                                }
                            }
                            OpenAIRealtimeMessage::ConversationItemCreate { item } => {
                                println!("New conversation item: {:?}", item);
                                if item.item_type == "function_call_output" {
                                    let mut parameter = MetadataParameters::default();
                                    parameter.insert(
                                        "tools".to_string(),
                                        dora_node_api::Parameter::String("[]".to_string()),
                                    );
                                    tx.send(DoraTokioBroadcast::Output(
                                        DataId::from("function_call_output".to_string()),
                                        parameter,
                                        item.output
                                            .clone()
                                            .unwrap_or_default()
                                            .into_arrow()
                                            .to_data(),
                                    ))
                                    .unwrap();
                                }
                            }
                            _ => {}
                        }
                    }
                    _ => break,
                }
                None
            }
            future::Either::Right(Err(_)) => break,
        };
        if let Some(frame) = frame {
            ws.write_frame(frame).await?;
        }
    }

    Ok(())
}

async fn server_upgrade(
    mut req: Request<Incoming>,
    tx: tokio::sync::broadcast::Sender<DoraTokioBroadcast>,
) -> Result<Response<Empty<Bytes>>, WebSocketError> {
    let (response, fut) = upgrade::upgrade(&mut req)?;

    tokio::task::spawn(async move {
        if let Err(e) = tokio::task::unconstrained(handle_client(fut, tx)).await {
            eprintln!("Error in websocket connection: {}", e);
        }
    });

    Ok(response)
}

pub fn lib_main() -> Result<(), WebSocketError> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .enable_time()
        .build()
        .unwrap();

    rt.block_on(async move {
        let port = std::env::var("PORT").unwrap_or_else(|_| "8123".to_string());
        let host = std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
        let addr = format!("{}:{}", host, port);
        let listener = TcpListener::bind(&addr).await?;
        println!("Server started, listening on {}", addr);
        let (mut node, mut events) = DoraNode::init_from_env().unwrap();

        let (tx, mut rx) = tokio::sync::broadcast::channel::<DoraTokioBroadcast>(16);
        let tx_dora = tx.clone();
        let dora_thread_handle = tokio::spawn(async move {
            loop {
                let event_fut = rx.recv().map(Either::Left);
                let frame_fut = events.recv_async().map(Either::Right);
                let event_stream = (event_fut, frame_fut).race();

                match event_stream.await {
                    futures_util::future::Either::Right(Some(Event::Input {
                        id,
                        metadata,
                        data,
                    })) => {
                        tx_dora
                            .send(DoraTokioBroadcast::Input(
                                id,
                                metadata.parameters,
                                data.into(),
                            ))
                            .unwrap();
                    }
                    futures_util::future::Either::Right(Some(Event::Stop(_))) => {
                        println!("Received stop event, shutting down.");
                        break;
                    }
                    futures_util::future::Either::Right(Some(_)) => {}
                    futures_util::future::Either::Right(None) => {
                        eprintln!("Error receiving event");
                        break;
                    }
                    futures_util::future::Either::Left(Ok(DoraTokioBroadcast::Output(
                        id,
                        metadata,
                        data,
                    ))) => {
                        if id != DataId::from("audio".to_string()) {
                            println!("Got the following output text: {}", id);
                        }
                        node.send_output(id, metadata, make_array(data)).unwrap();
                    }
                    futures_util::future::Either::Left(Ok(DoraTokioBroadcast::Input(
                        _id,
                        _metadata,
                        _data,
                    ))) => {}
                    futures_util::future::Either::Left(Err(_)) => {
                        eprintln!("Error receiving from channel");
                        break;
                    }
                }
            }
        });
        tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((stream, _)) => {
                        println!("Client connected");
                        let tx2 = tx.clone();
                        tokio::spawn(async move {
                            let io = hyper_util::rt::TokioIo::new(stream);
                            let conn_fut = http1::Builder::new()
                                .serve_connection(
                                    io,
                                    service_fn(move |req| server_upgrade(req, tx2.clone())),
                                )
                                .with_upgrades();
                            if let Err(e) = conn_fut.await {
                                println!("An error occurred: {:?}", e);
                            }
                        });
                    }
                    Err(e) => {
                        println!("Failed to accept connection: {:?}", e);
                    }
                }
            }
        });
        dora_thread_handle.await.unwrap();
        Ok(())
    })
}

#[cfg(feature = "python")]
use pyo3::{
    Bound, PyResult, Python, pyfunction, pymodule,
    types::{PyModule, PyModuleMethods},
    wrap_pyfunction,
};

#[cfg(feature = "python")]
#[pyfunction]
fn py_main(_py: Python) -> PyResult<()> {
    lib_main().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))
}

#[cfg(feature = "python")]
#[pymodule]
fn dora_openai_websocket(_py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_main, &m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
