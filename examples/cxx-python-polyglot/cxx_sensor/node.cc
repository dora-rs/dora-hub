#include "dora-node-api.h"
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    auto dora_node = init_dora_node();

    int tick_counter = 0;

    for (int i = 0; i < 50; i++) {
        auto event = dora_node.events->next();
        auto ty = event_type(event);

        if (ty == DoraEventType::AllInputsClosed) {
            break;
        } else if (ty == DoraEventType::Input) {
            tick_counter++;

            // 1. Generate an array of 100 float values (Simulating vibration data)
            std::vector<float> sensor_data(100);
            for (int j = 0; j < 100; j++) {
                sensor_data[j] = std::sin(j * 0.1f); // Normal baseline vibration
            }

            // 2. Inject a mechanical anomaly every 15 ticks!
            if (tick_counter % 15 == 0) {
                sensor_data[50] = 99.9f; // Massive vibration spike
            }

            // 3. The Zero-Copy  (Cast the float array to raw bytes)
            size_t byte_size = sensor_data.size() * sizeof(float);
            rust::Slice<const uint8_t> data_slice{
                reinterpret_cast<const uint8_t*>(sensor_data.data()), 
                byte_size
            };

            // 4. Send it instantly to the Python brain
            auto result = send_output(dora_node.send_output, "vibration_data", data_slice);
            
            auto error = std::string(result.error);
            if (!error.empty()) {
                std::cerr << "Error sending data: " << error << std::endl;
                return -1;
            }
        }
    }

    return 0;
}