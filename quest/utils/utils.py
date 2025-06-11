from enum import Enum

class TensorLayout(Enum):
    NHD = 0
    HND = 1

    FORMAT2STR = {0: "NHD", 1: "HND"}

# Enhanced Quest KV Cache Compression with Offloaded Token Support
"""
Quest Enhanced Offloaded Token Management

The enhanced Quest system now supports comprehensive offloading and consideration of 
all tokens (both prompt and previously decoded) during the decoding phase. 

Key Enhancements:
1. **Comprehensive Token Tracking**: The system now tracks prompt vs decoded tokens
2. **Full Offloading Support**: Both prompt and decoded tokens are properly offloaded
3. **Enhanced Attention Estimation**: All offloaded tokens are considered for future attention
4. **Dynamic Page Management**: Page budgets adapt to the actual available content

Usage Example:
```python
import quest.utils as quest_utils

# Initialize controller with enhanced tracking
controller = quest_utils.InferenceController(
    num_layers=32,
    num_heads=32, 
    head_dim=128,
    page_size=16,
    page_budget=64,
    max_seq_len=4096,
    dtype=torch.float16,
    device=torch.device("cuda")
)

# During prefill (prompt processing)
controller.prepare_metadata(prompt_length)  # Tracks prompt tokens
controller.begin_forward(prompt_length)
# ... process prompt with QuestAttention ...

# During decode (token generation)
for step in range(max_decode_steps):
    controller.prepare_metadata(1)  # Each decoded token is tracked
    controller.begin_forward(1)
    
    # The system now considers ALL previous tokens (prompt + decoded) 
    # for attention estimation and page selection
    # ... process with QuestAttention ...
    
    # Monitor offloaded token status
    info = quest_utils.get_offloaded_token_info(controller)
    print(f"Decoded tokens: {info['token_composition']['decoded_tokens_count']}")
    print(f"Available for attention: {info['attention_estimation']['pages_available_for_estimation']}")
    
    controller.end_forward()

controller.clean_states()
```

Key Behavioral Changes:
- **Prefill Phase**: Unchanged behavior, prompt tokens are offloaded as before
- **Decode Phase**: Now considers ALL historical pages (prompt + decoded tokens)
- **Attention Estimation**: Enhanced to work with dynamic page counts
- **Page Selection**: Top-K selection now includes all offloaded content
- **Memory Management**: Improved buffer allocation for variable page counts
"""
