use dora_message::{arrow_data::ArrayData, metadata::ArrowTypeInfo};
use dora_node_api::{self, DoraNode, Event, MetadataParameters};
use eyre::{Context, ContextCompat, bail};
use mcap::{Summary, WriteOptions, records::MessageHeader, write::Metadata};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    env,
    fs::File,
    ptr::NonNull,
    time::{Duration, Instant, SystemTime},
};

const METADATA_NAME: &str = "Dora Metadata";

#[derive(Clone, Serialize, Deserialize)]
pub struct RecordHeader {
    pub type_info: ArrowTypeInfo,
    pub parameters: MetadataParameters,
}

fn timestamp() -> Duration {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
}

fn run_passthrough() -> eyre::Result<()> {
    let (mut node, mut events) = DoraNode::init_from_env()?;
    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, data, metadata } => {
                node.send_output(id, metadata.parameters, data.0)?;
            }
            Event::Stop(_) => break,
            event => {
                eprintln!("Event: {event:#?}")
            }
        }
    }

    Ok(())
}

fn find_backing_buffer(
    data: &ArrayData,
    type_info: &ArrowTypeInfo,
) -> Option<(NonNull<u8>, usize)> {
    if let (Some(child_data), Some(child_type_info)) =
        (data.child_data().last(), type_info.child_data.last())
    {
        if let Some(result) = find_backing_buffer(child_data, child_type_info) {
            return Some(result);
        }
    }
    if let (Some(buffer), Some(offset)) = (data.buffers().last(), type_info.buffer_offsets.last()) {
        dbg!(offset);
        Some((buffer.data_ptr(), offset.offset + offset.len))
    } else {
        None
    }
}

fn run_record() -> eyre::Result<()> {
    let (_node, mut events) = DoraNode::init_from_env()?;
    let mut channels = HashMap::new();

    let mut writer = mcap::Writer::with_options(
        File::create(env::var("MCAP_FILE").unwrap_or_else(|_| "record.mcap".into()))?,
        WriteOptions::new(),
    )?;
    writer.write_metadata(&Metadata {
        name: METADATA_NAME.into(),
        metadata: [("start".into(), timestamp().as_millis().to_string())]
            .into_iter()
            .collect(),
    })?;
    let schema_id = writer.add_schema("Dora Record v1", "", &[])?;
    let mut sequence = 0;

    let mut buf = Vec::new();
    while let Some(event) = events.recv() {
        match event {
            Event::Input { id, data, metadata } => {
                use std::collections::hash_map::Entry;
                let channel_id = match channels.entry(id) {
                    Entry::Occupied(id) => *id.get(),
                    Entry::Vacant(entry) => {
                        let channel_id = writer.add_channel(
                            schema_id,
                            entry.key().as_str(),
                            "",
                            &BTreeMap::new(),
                        )?;
                        *entry.insert(channel_id)
                    }
                };

                let timestamp = metadata.timestamp();

                buf.clear();
                let header = RecordHeader {
                    type_info: metadata.type_info,
                    parameters: metadata.parameters,
                };
                bincode::serde::encode_into_std_write(
                    &header,
                    &mut buf,
                    bincode::config::standard(),
                )?;
                // In Dora, each Arrow data is backed by exactly one buffer.
                let (array_start, array_len) =
                    find_backing_buffer(&data.to_data(), &header.type_info)
                        .context("failed to find backing buffer")?;
                unsafe {
                    buf.extend_from_slice(std::slice::from_raw_parts(
                        array_start.as_ptr(),
                        array_len,
                    ));
                }

                let header = MessageHeader {
                    channel_id,
                    sequence,
                    log_time: SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .map_or(0, |it| it.as_millis() as u64),
                    publish_time: timestamp.get_time().to_duration().as_millis() as u64,
                };
                sequence += 1;

                writer.write_to_known_channel(&header, &buf)?;
                writer.flush()?;
            }
            event => {
                eprintln!("Event: {event:#?}")
            }
        }
    }

    Ok(())
}

fn run_replay() -> eyre::Result<()> {
    let (mut node, _events) = DoraNode::init_from_env()?;

    let fd = File::open(env::var("MCAP_FILE").unwrap_or_else(|_| "record.mcap".into()))
        .context("couldn't open MCAP file")?;
    let mapped = unsafe { Mmap::map(&fd) }.context("could't map MCAP file")?;
    if let Some(delay_ms) = env::var("DELAY_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
    {
        std::thread::sleep(Duration::from_millis(delay_ms));
    }

    let summary = Summary::read(&mapped)?.context("no summary found")?;
    let index = summary
        .metadata_indexes
        .iter()
        .find(|it| it.name == METADATA_NAME)
        .context("no Dora Metadata found in MCAP")?;
    let metadata = mcap::read::metadata(&mapped, index)?;
    let start = metadata
        .metadata
        .get("start")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or_default();
    let start_instant = Instant::now();

    let stream = mcap::MessageStream::new(&mapped)?;
    let mut schema_verified = false;

    let mut channel_enabled = vec![];
    for message in stream {
        let message = message?;
        if !schema_verified {
            let schema_version = message.channel.schema.as_ref().map(|s| s.name.as_str());
            if !schema_version.is_some_and(|s| s.ends_with("v1")) {
                bail!("Unsupported schema version: {:?}", schema_version);
            }
            schema_verified = true;
        }
        let channel_idx = message.channel.id as usize - 1;
        if channel_enabled.len() <= channel_idx {
            assert_eq!(channel_enabled.len(), channel_idx);
            let enabled = node.node_config().outputs.contains(&message.channel.topic);
            channel_enabled.push(enabled);
        }
        if !channel_enabled[channel_idx] {
            continue;
        }

        let publish_time =
            start_instant + Duration::from_millis(message.publish_time.saturating_sub(start));
        if let Some(delay) = publish_time.checked_duration_since(Instant::now()) {
            std::thread::sleep(delay);
        }

        let (header, header_len) = bincode::serde::decode_from_slice::<RecordHeader, _>(
            &message.data,
            bincode::config::standard(),
        )
        .context("couldn't decode record header")?;
        let data = &message.data[header_len..];
        node.send_typed_output(
            message.channel.topic.to_string().into(),
            header.type_info,
            header.parameters,
            data.len(),
            |dst| dst.copy_from_slice(data),
        )?;
    }

    Ok(())
}

fn main() -> eyre::Result<()> {
    let mode = env::var("MODE").unwrap_or_else(|_| "disable".into());
    match mode.as_str() {
        "disable" => run_passthrough(),
        "record" => run_record(),
        "replay" => run_replay(),
        other => bail!("Unknown mode: {other}"),
    }
}
