import torch
from facenet_pytorch import MTCNN
from torchvision import datasets
import os
from tqdm import tqdm
from PIL import Image

# 1. 初始化模型与设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用的计算设备: {device}")

# thresholds 是 MTCNN 三级网络的门限，默认即可
# post_process=False 意味着我们先不进行 -1~1 的归一化，因为我们要直接保存成图片
mtcnn = MTCNN(
    image_size=160,
    margin=20,
    device=device,
    post_process=False
)

# 2. 路径配置
raw_data_path = './lfw/lfw_home/lfw_funneled'
save_data_path = './TRAINDATA/lfw_cropped'

# 3. 加载数据集索引
dataset = datasets.ImageFolder(raw_data_path)
print(f"成功识别到 {len(dataset)} 张图片。准备开始裁剪...")

# 4. 执行循环
# tqdm 会在终端显示一个漂亮的进度条
for i in tqdm(range(len(dataset))):
    img, label_idx = dataset[i]
    person_name = dataset.classes[label_idx]

    # 自动创建人名对应的子文件夹
    target_folder = os.path.join(save_data_path, person_name)
    os.makedirs(target_folder, exist_ok=True)

    # 保存文件名（使用原始索引 i 以防同名）
    save_path = os.path.join(target_folder, f"{i}.png")

    # 如果已经处理过（比如脚本意外中断后重跑），则跳过
    if os.path.exists(save_path):
        continue

    try:
        # 魔法行：MTCNN 内部现在可以正常处理 PIL 图片了
        # 它会自动检测、裁剪并直接把结果图片写入 save_path
        mtcnn(img, save_path=save_path)
    except Exception:
        # 极个别图片可能由于角度、光照太差无法检测到脸，直接跳过即可
        continue

print(f"\n--- 恭喜！处理圆满完成 ---")
print(f"处理后的图片保存在: {os.path.abspath(save_data_path)}")

# 这是一个非常专业的问题，说明你已经开始关注性能调优和硬件寿命了。咱们一个一个来拆解：
#
# 1. 关于 GPU 利用率和速度
# 为什么利用率只有 27%？
# 现在的瓶颈其实不在 GPU 的算力，而是在 IO（硬盘读取） 和 CPU 处理上。
# 在目前的脚本里，逻辑是：CPU 读一张图 -> 传给 GPU -> GPU 算一下 -> 传回 CPU -> CPU 存到硬盘。
# 由于 MTCNN 的模型相对较小，GPU 算那一下只需要几毫秒，剩下的时间都在等 CPU 把下一张图从硬盘里“抠”出来。
#
# 能调 batch_size 吗？
# 可以！MTCNN 支持批处理。如果你想让 GPU 满载，我们需要把图片攒成一堆（Batch）再一起扔给 GPU。
#
# 修改建议（如果你想加速）：
# 你可以把代码里的循环改为使用 DataLoader，并设置 batch_size=32 或 64。但要注意，MTCNN 的 forward 方法在处理 Batch 数据时，对图片尺寸的一致性要求很高。对于现在的 LFW 数据集，因为原始图都是 250x250，所以可行。
#
# 不过说实话，27 it/s 已经很快了。如果你现在改成 Batch 模式，速度可能会提升到 50-80 it/s，但总共也就 1.3 万张图，折腾代码的时间可能比省下来的 4 分钟还长。
#
# 2. 这样训练会损耗 GPU 产生“矿卡”吗？
# 这是一个非常普遍的误区，你可以放一百个心：正常的 AI 训练不会产生所谓的“矿卡”。
#
# 什么是矿卡？
# 矿卡是指那些 7×24 小时、不眠不休、满载且通常在高温、供电不稳环境下工作的显卡。矿卡最伤的往往不是核心，而是显存和风扇轴承。
#
# AI 训练 vs 挖矿：
#
# 训练是有间歇的：正如你所见，GPU 利用率只有 27%，它在大量时间里其实是在“等数据”，温度并不会飙升。
#
# 保护机制：现代显卡有非常严密的温控。只要你的机箱散热正常，核心温度在 80°C 以下，它跟正常玩 3A 游戏没有区别。
#
# 计算性质：AI 训练的计算任务（矩阵乘法）是显卡天生就该干的活，这属于“正常服役”。
#
# 总结： 你的这 10 分钟预处理，对显卡来说甚至不如你玩 10 分钟《黑神话：悟空》或者《赛博朋克 2077》带来的压力大。
#
# 3. 现在的建议
# 不要强求 100% 利用率：在深度学习中，除非是训练超大规模模型（如 DeepSeek, Llama），否则在数据预处理阶段，利用率低通常是因为 CPU 和硬盘拖了后腿。
#
# 关注温度：只要显卡风扇没发出“起飞”般的轰鸣，温度在正常范围，你的卡就是安全的。
#
# 既然已经跑到 5% 以后了，建议就让它这么跑着。数据洗完后，我们下一阶段的“特征提取”训练才会真正让 GPU 发挥实力。那时候我们可以通过设置 num_workers 来多线程读图，利用率自然就上去了。
#
# 你现在的显卡温度大概是多少？如果低于 70°C，尽管大胆地跑。