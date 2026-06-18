# dora-recorder

dora data recording in Apache Arrow format

## Adding to existing graph

```yaml
- id: dora-recorder
  path: path-to-dora-recorder/target/release/dora-recorder
  inputs:
    image: webcam/image
    lidar: lidar/pointcloud
    # ... (any input which is going to be logged)
```

## Output Files

**Format**: Arrow file
**Path**: `record-bag/type-<TYPE_ID>.parquet` (TYPE-ID starts from 1 and ends at the total number of `data_types` of the message to be recorded)

**Columns**:
- payload_list: List, containing the input messages
- timestamp: u64, representing the timestamp in [Unique Hybrid Logical Clock time](https://github.com/atolab/uhlc-rs)
- topic: id

Example:

```json
------------------------------------------------
| payload_list |      timestamp      |  topic  |
------------------------------------------------
| <Arrow Msg>  | 7651893987555135264 | x-topic |
------------------------------------------------
| <Arrow Msg>  | 7651894009906043616 | y-topic |
------------------------------------------------
...
```
