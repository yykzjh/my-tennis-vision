import tensorrt as trt

# 创建一个logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# 构建引擎
def build_engine(onnx_file_path, engine_file_path):
    # 创建builder和network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 解析ONNX模型
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # 构建优化的引擎
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 20  # 这个值根据您的模型大小和可用内存来调整
    engine = builder.build_engine(network, config)

    # 保存引擎到文件
    with open(engine_file_path, 'wb') as f:
        f.write(engine.serialize())

    return engine

# 指定ONNX模型路径和目标TensorRT引擎文件路径
onnx_file_path = './pretrain/action_classify.onnx'
engine_file_path = 'pretrain/action_classify.engine'

# 转换模型
build_engine(onnx_file_path, engine_file_path)
