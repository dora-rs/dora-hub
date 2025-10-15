use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

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
pub enum ContentPart {
    #[serde(rename = "input_text")]
    InputText { text: String },
    #[serde(rename = "input_audio")]
    InputAudio {
        audio: String,
        transcript: Option<String>,
    },
    #[serde(rename = "input_image")]
    InputImage {
        #[serde(rename = "type")]
        ty: String,
        image_url: String,
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
