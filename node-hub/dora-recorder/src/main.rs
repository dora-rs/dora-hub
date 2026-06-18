use arrow::array::{Array, ArrayRef, ListArray, StringArray, UInt64Array};
use arrow::buffer::OffsetBuffer;
use arrow::compute::concat;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::FileWriter;
use arrow::record_batch::RecordBatch;
use dora_node_api::{self, DoraNode, Event};
use std::collections::HashMap;
use std::fs::{File, create_dir_all};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::mpsc::{UnboundedSender, unbounded_channel};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::task::{JoinHandle, spawn};

// Wrapped message, allowing the permit to flow in the channel along with the message
struct RecordMessage {
    data: (ArrayRef, u64, String),
    size: usize,
    _permit: OwnedSemaphorePermit,
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let (_node, mut events) = DoraNode::init_from_env()?;
    let buffer_size = 16 * 1024 * 1024;
    let flush_threshold = buffer_size / 4;

    // map structure: data_type --> asynchronous write task
    let mut type_channels: HashMap<
        DataType,
        (
            UnboundedSender<RecordMessage>,
            Arc<Semaphore>, // 每个通道独立的信号量
            Arc<AtomicUsize>,
            JoinHandle<()>,
        ),
    > = HashMap::new();

    let record_path = "record-bag";
    let mut type_count = 0;
    create_dir_all(record_path)?;

    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, data, metadata } => {
                let msg_type_clone = data.data_type().clone();
                let msg_length_clone = data.to_data().get_array_memory_size();
                let msg_timestamp = metadata.timestamp().get_time().as_u64();
                let msg_topic = id.to_string();

                // Find the corresponding sender based on data_type
                // If not found, execute the function creating asynchronous write task
                let (sender, semaphore, _, _) = type_channels
                    .entry(msg_type_clone.clone())
                    .or_insert_with(|| {
                        type_count += 1;

                        let (tx, mut rx) = unbounded_channel::<RecordMessage>();
                        let semaphore = Arc::new(Semaphore::new(buffer_size));
                        let msg_counter = Arc::new(AtomicUsize::new(0));
                        let msg_counter_clone = msg_counter.clone();
                        let file_name = format!("{}/type-{}.arrow", record_path, type_count);
                        let file = File::create(&file_name).expect("Failed to create file");
                        println!("Creating new recording file: {}", file_name);

                        // create apache arrow schema
                        let payload_field =
                            Arc::new(Field::new("item", msg_type_clone.clone(), true));
                        let list_field = Arc::new(Field::new(
                            "payload_list",
                            DataType::List(payload_field.clone()),
                            false,
                        ));
                        let timestamp_field =
                            Arc::new(Field::new("timestamp", DataType::UInt64, false));
                        let topic_field = Arc::new(Field::new("topic", DataType::Utf8, false));
                        let schema =
                            Arc::new(Schema::new(vec![list_field, timestamp_field, topic_field]));

                        let mut writer = FileWriter::try_new(file, &schema)
                            .expect("Failed to create file writer");

                        let handle = spawn(async move {
                            let mut payload_vec: Vec<ArrayRef> = Vec::new();
                            let mut timestamp_vec: Vec<u64> = Vec::new();
                            let mut topic_vec: Vec<String> = Vec::new();
                            let mut permit_vec: Vec<OwnedSemaphorePermit> = Vec::new();
                            let mut current_batch_memory = 0;

                            let mut save_batch =
                                |payloads: &mut Vec<ArrayRef>,
                                 timestamps: &mut Vec<u64>,
                                 topics: &mut Vec<String>,
                                 permits: &mut Vec<OwnedSemaphorePermit>,
                                 current_batch_memory: &mut usize| {
                                    if payloads.is_empty() {
                                        return;
                                    }

                                    let current_batch_size = timestamps.len();
                                    let mut offsets = Vec::with_capacity(payloads.len() + 1);
                                    let mut current_offset = 0;
                                    offsets.push(current_offset);

                                    for arr in payloads.into_iter() {
                                        current_offset += arr.len() as i32;
                                        offsets.push(current_offset);
                                    }

                                    let offset_buffer = OffsetBuffer::new(offsets.into());
                                    let array_refs: Vec<&dyn Array> =
                                        payloads.iter().map(|a| a.as_ref()).collect();
                                    let values_array = concat(&array_refs).unwrap();
                                    let list_array = ListArray::new(
                                        payload_field.clone(),
                                        offset_buffer,
                                        values_array,
                                        None,
                                    );

                                    let batch = RecordBatch::try_new(
                                        schema.clone(),
                                        vec![
                                            Arc::new(list_array),
                                            Arc::new(UInt64Array::from(timestamps.clone())),
                                            Arc::new(StringArray::from(topics.clone())),
                                        ],
                                    )
                                    .unwrap();

                                    writer
                                        .write(&batch)
                                        .expect("Error occurred in writing a batch");
                                    *current_batch_memory = 0;
                                    payloads.clear();
                                    timestamps.clear();
                                    topics.clear();
                                    permits.clear();

                                    msg_counter_clone
                                        .fetch_add(current_batch_size, Ordering::Relaxed);
                                };

                            while let Some(msg_wrap) = rx.recv().await {
                                current_batch_memory += msg_wrap.size;
                                payload_vec.push(msg_wrap.data.0);
                                timestamp_vec.push(msg_wrap.data.1);
                                topic_vec.push(msg_wrap.data.2);
                                permit_vec.push(msg_wrap._permit);

                                if current_batch_memory >= flush_threshold {
                                    save_batch(
                                        &mut payload_vec,
                                        &mut timestamp_vec,
                                        &mut topic_vec,
                                        &mut permit_vec,
                                        &mut current_batch_memory,
                                    );
                                }
                            }

                            // graceful shutdown
                            println!(
                                "Channel closed, writing remaining data: {:?} ({} left)",
                                msg_type_clone,
                                payload_vec.len()
                            );

                            if !payload_vec.is_empty() {
                                save_batch(
                                    &mut payload_vec,
                                    &mut timestamp_vec,
                                    &mut topic_vec,
                                    &mut permit_vec,
                                    &mut current_batch_memory,
                                );
                            }

                            // Explicitly end the writer, ensure that the footer writes and refreshes the file
                            if let Err(e) = writer.finish() {
                                eprintln!("Failed to close writer: {}", e);
                            }
                        });

                        (tx, semaphore, msg_counter, handle)
                    });

                // try to acquire the same semaphore as the memory size occupied by the message (in bytes)
                match semaphore
                    .clone()
                    .try_acquire_many_owned(msg_length_clone as u32)
                {
                    Ok(permit) => {
                        let msg_wrap = RecordMessage {
                            data: (data.to_owned(), msg_timestamp, msg_topic),
                            size: msg_length_clone,
                            _permit: permit,
                        };

                        if let Err(_) = sender.send(msg_wrap) {
                            // eprintln!("Sending data failed (the receiving end may have been closed): {}", e);
                        };
                    }
                    Err(_) => {
                        // This channel is full and will be discarded directly
                        // without affecting other data types or blocking the main loop
                        // eprintln!("Channel for {:?} is full, dropping message", msg_type_clone);
                    }
                }
            }
            Event::InputClosed { id } => {
                println!("Input '{id}' is closed.");
            }
            Event::Error(err) => {
                println!("Dora event stream error: {err}");
            }
            _event => {
                println!("Stop recording...");

                // Drop all the senders by taking ownership of the map and consuming it
                let mut handles = Vec::new();

                for (data_type, (sender, _semaphore, saved_msg_count, handle)) in
                    type_channels.drain()
                {
                    drop(sender);
                    handles.push((data_type, saved_msg_count, handle));
                }

                println!("Waiting for all writing tasks to complete...");

                for (data_type, _, handle) in handles {
                    if let Err(e) = handle.await {
                        eprintln!("task {:?} wait failed: {}", data_type, e);
                    }
                }

                println!("All data has been written in");
                break;
            }
        }
    }

    Ok(())
}
