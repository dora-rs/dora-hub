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
use eyre::Context;
use eyre::Result;
use fastwebsockets::Frame;
use fastwebsockets::OpCode;
use fastwebsockets::Payload;
use fastwebsockets::WebSocketError;
use fastwebsockets::upgrade;
use futures_concurrency::future::Race;
use futures_util::FutureExt;
use futures_util::future;
use futures_util::future::Either;
use http_body_util::BodyExt;
use hyper::Request;
use hyper::Response;
use hyper::body::Bytes;
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use sha1::Digest;
use tokio::net::TcpListener;
mod message;
mod model;
mod realtime;
use crate::message::ChatCompletionObject;
use crate::message::ChatCompletionObjectChoice;
use crate::message::ChatCompletionObjectMessage;
use crate::message::ChatCompletionRequest;
use crate::message::FinishReason;
use crate::message::Usage;
use crate::model::ListModelsResponse;
use crate::model::Model;
use crate::realtime::ConversationItem;
use crate::realtime::OpenAIRealtimeMessage;
use crate::realtime::OpenAIRealtimeResponse;
use crate::realtime::ResponseDoneData;
use crate::realtime::ToolCall;
use tokio::sync::broadcast;

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
enum BroadcastMessage {
    Output(DataId, MetadataParameters, ArrayData),
    Input(DataId, MetadataParameters, ArrayRef),
}

async fn handle_client(
    fut: upgrade::UpgradeFut,
    tx: tokio::sync::broadcast::Sender<BroadcastMessage>,
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

    tx.send(BroadcastMessage::Output(
        DataId::from("system_prompt".to_string()),
        MetadataParameters::default(),
        system_prompt.into_arrow().to_data(),
    ))
    .unwrap();
    tx.send(BroadcastMessage::Output(
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
            future::Either::Left(Ok(BroadcastMessage::Input(id, _metadata, data))) => {
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
            future::Either::Left(Ok(BroadcastMessage::Output(_, _, _))) => {
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
                                    tx.send(BroadcastMessage::Output(
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
                                    tx.send(BroadcastMessage::Output(
                                        DataId::from("response.create".to_string()),
                                        parameter,
                                        text.into_arrow().to_data(),
                                    ))
                                    .unwrap();
                                }
                            }
                            OpenAIRealtimeMessage::ConversationItemCreate { item } => {
                                println!("New conversation item: {:?}", item);
                                match item.item_type.as_str() {
                                    "function_call_output" => {
                                        let mut parameter = MetadataParameters::default();
                                        parameter.insert(
                                            "tools".to_string(),
                                            dora_node_api::Parameter::String("[]".to_string()),
                                        );
                                        tx.send(BroadcastMessage::Output(
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
                                    "message" => {
                                        let contents = item.content;
                                        let texts: Vec<String> = contents
                                            .iter()
                                            .filter_map(|part| match part {
                                                realtime::ContentPart::Text { text } => {
                                                    Some(text.clone())
                                                }
                                                realtime::ContentPart::InputText { text } => {
                                                    Some(text.clone())
                                                }
                                                realtime::ContentPart::InputImage {
                                                    image_url,
                                                    ..
                                                } => Some(format!(
                                                    "<|user|>\n<|vision_start|>\n{}",
                                                    image_url
                                                )),
                                                _ => None,
                                            })
                                            .collect();
                                        tx.send(BroadcastMessage::Output(
                                            DataId::from("text".to_string()),
                                            MetadataParameters::default(),
                                            texts.into_arrow().to_data(),
                                        ))
                                        .unwrap();
                                    }
                                    _ => {}
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

/// List all models available.
pub(crate) fn models_handler() -> Result<Response<http_body_util::Full<Bytes>>> {
    // log
    let custom_model = Model {
        id: "custom_model".to_string(),
        created: 123,
        object: "model".to_string(),
        owned_by: "dora".to_string(),
    };
    let list_models = vec![custom_model];

    let list_models_response = ListModelsResponse {
        object: "list".to_string(),
        data: list_models,
    };
    // serialize response
    let s = serde_json::to_string(&list_models_response).context("Failed to serialize response")?;

    // return response
    Response::builder()
        .header("Access-Control-Allow-Origin", "*")
        .header("Access-Control-Allow-Methods", "*")
        .header("Access-Control-Allow-Headers", "*")
        .header("Content-Type", "application/json")
        .body(s.into())
        .context("Failed to build response")
}

fn sec_websocket_protocol(key: &[u8]) -> String {
    use base64::engine::general_purpose::STANDARD;
    use sha1::Sha1;
    let mut sha1 = Sha1::new();
    sha1.update(key);
    sha1.update(b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"); // magic string
    let result = sha1.finalize();
    STANDARD.encode(&result[..])
}

async fn server_upgrade(
    mut req: Request<Incoming>,
    tx: tokio::sync::broadcast::Sender<BroadcastMessage>,
) -> Result<Response<http_body_util::Full<Bytes>>> {
    match req.uri().path() {
        "/realtime" | "/v1/realtime" => {
            let (_response, fut) = upgrade::upgrade(&mut req)?;

            tokio::task::spawn(async move {
                if let Err(e) = tokio::task::unconstrained(handle_client(fut, tx)).await {
                    eprintln!("Error in websocket connection: {}", e);
                }
            });
            let key = req
                .headers()
                .get("Sec-WebSocket-Key")
                .ok_or(WebSocketError::MissingSecWebSocketKey)?;
            let response = Response::builder()
                .status(hyper::StatusCode::SWITCHING_PROTOCOLS)
                .header(hyper::header::CONNECTION, "upgrade")
                .header(hyper::header::UPGRADE, "websocket")
                .header(
                    "Sec-WebSocket-Accept",
                    &sec_websocket_protocol(key.as_bytes()),
                )
                .body("123".into())
                .expect("bug: failed to build response");

            Ok(response)
        }
        "/v1/chat/completions" => chat_completions_handler(req, tx).await,
        "/models" | "/v1/models" => models_handler(),
        _ => {
            let response = Response::builder()
                .status(hyper::StatusCode::NOT_FOUND)
                .body("Not found".into())
                .expect("bug: failed to build response");
            print!("Unknown path: {}", req.uri().path());
            Ok(response)
        }
    }
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

        let (tx, mut rx) = tokio::sync::broadcast::channel::<BroadcastMessage>(16);
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
                            .send(BroadcastMessage::Input(
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
                    futures_util::future::Either::Left(Ok(BroadcastMessage::Output(
                        id,
                        metadata,
                        data,
                    ))) => {
                        if id != DataId::from("audio".to_string()) {
                            println!("Got the following output text: {}", id);
                        }
                        node.send_output(id, metadata, make_array(data)).unwrap();
                    }
                    futures_util::future::Either::Left(Ok(BroadcastMessage::Input(
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

// Forked from https://github.com/LlamaEdge/LlamaEdge/blob/6bfe9c12c85bf390c47d6065686caeca700feffa/llama-api-server/src/backend/ggml.rs#L301
async fn chat_completions_handler(
    req: Request<Incoming>,
    request_tx: broadcast::Sender<BroadcastMessage>,
) -> Result<Response<http_body_util::Full<Bytes>>> {
    if req.method().eq(&hyper::http::Method::OPTIONS) {
        let result = Response::builder()
            .header("Access-Control-Allow-Origin", "*")
            .header("Access-Control-Allow-Methods", "*")
            .header("Access-Control-Allow-Headers", "*")
            .header("Content-Type", "application/json")
            .body("".into())
            .context("Failed to build response")?;
        return Ok(result);
    }
    // parse request
    let body_bytes = req.collect().await.context("Failed to read request body")?;
    let chat_request: ChatCompletionRequest = serde_json::from_slice(&body_bytes.to_bytes())
        .context("Failed to deserialize chat completion request")?;

    let mut rx = request_tx.subscribe();
    request_tx
        .send(BroadcastMessage::Output(
            DataId::from("response.create".to_string()),
            MetadataParameters::default(),
            chat_request.to_texts().into_arrow().to_data(),
        ))
        .context("failed to send request")?;
    let res = loop {
        match rx.recv().await {
            Ok(BroadcastMessage::Input(id, _metadata, data)) => {
                let s = if data.data_type() == &DataType::Utf8 && id.contains("text") {
                    let data = data.as_string::<i32>();
                    let str = data.iter().fold("".to_owned(), |mut acc, x| {
                        if let Some(x) = x {
                            acc.push('\n');
                            acc.push_str(x);
                        }
                        acc
                    });
                    str.to_owned()
                } else {
                    "".to_owned()
                };
                let chat_completion_object = ChatCompletionObject {
                    id: "123".to_string(),
                    object: "chat.completion".to_string(),
                    created: 123,
                    model: "123".to_string(),
                    choices: vec![ChatCompletionObjectChoice {
                        index: 0,
                        message: ChatCompletionObjectMessage {
                            content: Some(s),
                            role: message::ChatCompletionRole::Assistant,
                            tool_calls: vec![],
                            function_call: None,
                        },
                        finish_reason: FinishReason::stop,
                        logprobs: None,
                    }],
                    usage: Usage {
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        total_tokens: 0,
                    },
                };
                let s = serde_json::to_string(&chat_completion_object)
                    .context("Failed to serialize response")?;
                // return response
                let result = Response::builder()
                    .header("Access-Control-Allow-Origin", "*")
                    .header("Access-Control-Allow-Methods", "*")
                    .header("Access-Control-Allow-Headers", "*")
                    .header("Content-Type", "application/json")
                    .body(s.into());
                break result;
            }
            _ => {}
        }
    };

    res.context("Failed to build response")
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
