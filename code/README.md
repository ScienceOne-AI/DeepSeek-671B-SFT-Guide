我们对 xtuner 框架做了如下修改。

1. `/model/transformers_models/deepseek_v3`: 新增 deepseek v3 的 modeling、configuration、tokenization（fast）。
2. `xtuner/utils/templates.py`: 添加了 deepseek_v3 的 prompt template，以适配 DeepSeek V3 和 R1 的训练模版。
3. `xtuner/model/sft.py`: 添加了对 DeepSeek V3 系列模型的 Flash Attention 2 的支持。
4. `xtuner/model/modules/dispatch/deepseek_v3.py`: 新增对 DeepSeek V3 的支持。
4. `xtuner/model/transformers_models/__init__.py`: 新增对 DeepSeek V3 的支持。
6. `xtuner/model/modules/dispatch/__init__.py`: 新增对 DeepSeek V3 的支持。
7. `xtuner/utils/handle_moe_load_and_save.py`: 新增对 DeepSeek V3 的支持。
8. `xtuner/utils/zero_to_any_dtype.py`: PyTorch 2.6 及以上版本默认使用 weights_only=True 来加载 checkpoint，但 .pth 文件包含了一些 DeepSpeed Zero Optimizer 的状态信息，导致无法直接反序列化。这里修改代码从而允许 ConfigDict、HistoryBuffer 反序列化。


----

We have made the following modifications to the xtuner framework.

1. `/model/transformers_models/deepseek_v3`: Added modeling, configuration, and tokenization (fast) for deepseek v3.
2. `xtuner/utils/templates.py`: Added prompt template for deepseek_v3 to adapt to the training templates of DeepSeek V3 and R1.
3. `xtuner/model/sft.py`: Added support for Flash Attention 2 for the DeepSeek V3 series models.
4. `xtuner/model/modules/dispatch/deepseek_v3.py`: Added support for DeepSeek V3.
5. `xtuner/model/transformers_models/__init__.py`: Added support for DeepSeek V3.
6. `xtuner/model/modules/dispatch/__init__.py`: Added support for DeepSeek V3.
7. `xtuner/utils/handle_moe_load_and_save.py`: Added support for DeepSeek V3.
8. `xtuner/utils/zero_to_any_dtype.py`: PyTorch 2.6 and above versions use weights_only=True by default to load checkpoints, but .pth files contain some DeepSpeed Zero Optimizer state information, making it impossible to deserialize directly. Modified the code here to allow deserialization of ConfigDict and HistoryBuffer.
