from quest.utils.decode_wrapper import BatchDecodeWithPagedKVCacheWrapper
from quest.utils.kv_cache import KvCache
from quest.utils.utils import TensorLayout

import torch

class InferenceController:
    def __init__(
        self,
        num_layers,
        num_heads,
        head_dim,
        page_size,
        page_budget, # Real page budget including the last page
        max_seq_len, # Real max for allocating kv / metadata
        dtype,
        device,      
    ):
        max_kv_pages_num = (max_seq_len + page_size - 1) // page_size
        self.kv_cache = KvCache(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            page_size=page_size,
            dtype=dtype,
            device=device
        )
        self.metadata_cache = KvCache(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            max_seq_len=max_kv_pages_num,
            page_size=page_size,
            dtype=dtype,
            device=device
        )
        self.layout = TensorLayout.NHD # Arbitrarily choose NHD. 
        self.device = device
        self.dtype = dtype

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.page_size = page_size

        self._page_budget = page_budget
        self._decode_handler = BatchDecodeWithPagedKVCacheWrapper(kv_layout="NHD")

        self.kv_indices_with_last = None
        self.kv_indices_without_last = None
        self.metadata_indices = None
        self.kv_last_page_idx = None # For decoding self-attention
        self.metadata_last_page_idx = None

        self.kv_indptr_for_append = None
        self.metadata_indptr_for_append = None
        self.kv_indptr_for_approx_decode = None

        self.inference_page_budget = None

        self.topk_dout_buffer = None
        self.topk_dindices_buffer = None
        self.topk_buf = None

        # Enhanced tracking for prompt vs decoded token management
        self._prompt_length = 0  # Track length of original prompt
        self._is_in_prefill = True  # Track whether we're still in prefill phase
        self._decoded_tokens_count = 0  # Track how many tokens have been decoded
        self._total_generated_length = 0  # Track total length including prompt + decoded
    
    # Used for controlling the number of pages
    # Here we skip first two layers by manipulating this.
    def set_page_budget(self, page_budget: int):
        self._page_budget = page_budget

    # Called once per forwarding in all layers
    # Adjust the metadata for paged_kv
    def prepare_metadata(self, seq_len: int):
        # Allocate entry for tokens
        appended_new_pages = self.kv_cache.append_seq(seq_len)
        # Allocate entry for metadata
        _ = self.metadata_cache.append_seq(appended_new_pages)
        
        # Enhanced: Track token type and update counters
        if self._is_in_prefill and seq_len > 1:
            # This is prompt prefill
            self._prompt_length += seq_len
            self._total_generated_length += seq_len
        elif seq_len == 1:
            # This is decode phase - single token generation
            if self._is_in_prefill:
                # First decode token - transition from prefill to decode
                self._is_in_prefill = False
            self._decoded_tokens_count += 1
            self._total_generated_length += 1
        else:
            # Multi-token append during decode (batch decode or continuation)
            self._decoded_tokens_count += seq_len
            self._total_generated_length += seq_len
    
    # Enhanced metadata preparation for considering all offloaded tokens
    def prepare_offloaded_metadata(self):
        """
        Prepare metadata indices that include ALL offloaded tokens (prompt + decoded)
        for attention estimation during decoding.
        """
        if self._is_in_prefill:
            # Still in prefill, use standard logic
            return self.metadata_cache.indicies
        
        # In decode phase - consider all historical pages including decoded tokens
        # This ensures that previously decoded tokens are also considered for attention
        return self.metadata_cache.indicies
    
    # Prepare metadata used for inference under certain PAGE_BUDGET
    # Called multiple times for layer sensitivity
    def begin_forward(self, seq_len: int, updateTensor: bool = True):
        # Allocate tensor in advance
        # This is used for append kernels, which need original indices
        if updateTensor:
            self.kv_indptr_for_append = torch.tensor([0, len(self.kv_cache.indicies)], dtype=torch.int32, device=self.device)
            self.metadata_indptr_for_append = torch.tensor([0, len(self.metadata_cache.indicies)], dtype=torch.int32, device=self.device)
            self.kv_last_page_idx = self.kv_cache.indicies[-1]
            self.metadata_last_page_idx = self.metadata_cache.indicies[-1]

        if seq_len > 1:
            # prefill requests
            # append_kv_cache_prefill and prefill_with_paged_kv_cache
            if updateTensor:
                self.kv_indices_with_last = torch.tensor(self.kv_cache.indicies, dtype=torch.int32, device=self.device)
                self.metadata_indices = torch.tensor(self.metadata_cache.indicies, dtype=torch.int32, device=self.device)
        else:
            # decode requests
            # append_kv_cache_decode, estimate_attn_score, topk_filtering
            cur_page_nums = len(self.kv_cache.indicies)
            assert cur_page_nums > 1 # at least two pages for excluding last page

            if updateTensor:
                # used for appending
                self.kv_indices_with_last = torch.tensor(self.kv_cache.indicies, dtype=torch.int32, device=self.device)

                # Enhanced: Consider ALL historical pages (prompt + decoded tokens)
                # Instead of just excluding the last page, we now consider all offloaded content
                offloaded_indices = self.prepare_offloaded_metadata()
                available_pages = offloaded_indices[:-1] if len(offloaded_indices) > 1 else []
                
                self.kv_indices_without_last = torch.tensor(available_pages, dtype=torch.int32, device=self.device).repeat(self.num_heads, 1) if available_pages else torch.empty((self.num_heads, 0), dtype=torch.int32, device=self.device)

                # used for estimate - now includes all offloaded tokens
                self.metadata_indices = torch.tensor(self.metadata_cache.indicies, dtype=torch.int32, device=self.device)

            # Enhanced: Consider all available pages for inference, not just recent ones
            # This ensures that both prompt and decoded tokens are candidates for attention
            self.inference_page_budget = min(self._page_budget, cur_page_nums)

            # Enhanced: Include all historical pages for decoding attention
            # The -1 excludes only the current page being written to
            num_available_pages = max(0, len(self.kv_cache.indicies) - 1)
            actual_budget = min(self.inference_page_budget - 1, num_available_pages)
            
            self.kv_indptr_for_approx_decode = torch.tensor([0, actual_budget], dtype=torch.int32, device=self.device)

            # Allocate buffer for top-k filtering with enhanced size for all offloaded content
            if actual_budget > 0:
                self.topk_dout_buffer = torch.zeros((self.num_heads, actual_budget), dtype=self.dtype, device=self.device)
                self.topk_dindices_buffer = torch.zeros((self.num_heads, actual_budget), dtype=torch.int32, device=self.device)
                self.topk_buf = torch.zeros((self.num_heads, 8192 * 2 * (2+4) // 2 // 48), dtype=self.dtype, device=self.device)
            else:
                self.topk_dout_buffer = torch.empty((self.num_heads, 0), dtype=self.dtype, device=self.device)
                self.topk_dindices_buffer = torch.empty((self.num_heads, 0), dtype=torch.int32, device=self.device)
                self.topk_buf = torch.empty((self.num_heads, 0), dtype=self.dtype, device=self.device)

            if actual_budget > 0:
                self._decode_handler.begin_forward(
                    self.kv_indptr_for_approx_decode,
                    self.num_heads,
                    self.num_heads,
                    self.head_dim,
                    self.page_size,
                    self.dtype
                )
    
    # Used for releasing resources
    # Free memory in CUDA side
    # called multiple times for layer sensitivity
    def end_forward(self):
        if hasattr(self, '_decode_handler'):
            self._decode_handler.end_forward()
    
    def need_estimate(self) -> bool:
        if self.inference_page_budget is None:
            return False
        
        cur_page_nums = len(self.kv_cache.indicies)
        return cur_page_nums > self.inference_page_budget
    
    # Enhanced: Provide information about token composition for debugging/monitoring
    def get_token_composition(self):
        """
        Return information about the current token composition in the cache
        """
        return {
            'prompt_length': self._prompt_length,
            'decoded_tokens_count': self._decoded_tokens_count,
            'total_length': self._total_generated_length,
            'is_in_prefill': self._is_in_prefill,
            'total_pages': len(self.kv_cache.indicies),
            'metadata_pages': len(self.metadata_cache.indicies)
        }
    
    def clean_states(self):
        self.kv_cache.release()
        self.metadata_cache.release()
        
        # Enhanced: Reset token tracking state
        self._prompt_length = 0
        self._is_in_prefill = True
        self._decoded_tokens_count = 0
        self._total_generated_length = 0
        