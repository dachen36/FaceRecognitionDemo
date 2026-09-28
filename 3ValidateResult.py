import os

# 指向你刚才生成的那个“洗干净”的文件夹
processed_data_path = './TRAINDATA/lfw_cropped'

# 获取所有子文件夹（人名）
people = [d for d in os.listdir(processed_data_path) if os.path.isdir(os.path.join(processed_data_path, d))]

print(f"统计结果：")
print(f"--- 成功提取了 {len(people)} 个人的面部数据 ---")

# 找一个照片比较多的人看看（比如布什，他是 LFW 里的常客）
bush_path = os.path.join(processed_data_path, 'George_W_Bush')
if os.path.exists(bush_path):
    pics = os.listdir(bush_path)
    print(f"George_W_Bush 文件夹下共有 {len(pics)} 张对齐后的脸部图片。")