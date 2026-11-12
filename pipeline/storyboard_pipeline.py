"""
AK-AIGC-Studio: 端到端多模态故事板流水线
剧本 -> 语音合成 -> 关键帧生图 -> 动态视频生成 -> 音画对齐输出
"""
import os
import argparse
from audio.infer_tts import synthesize_speech
from image.generate_image import generate_image
from video.generate_video import generate_video

def run_storyboard_pipeline(script: str, image_prompt: str, video_motion_prompt: str, output_name: str = "scene_01"):
    output_dir = os.path.join("outputs", output_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n==========================================")
    print(f"🎬 [Pipeline] 启动端到端故事板合成: {output_name}")
    print(f"==========================================")
    
    # 步骤 1: 音频生成
    audio_path = os.path.join(output_dir, "speech.wav")
    synthesize_speech(text=script, output_path=audio_path)
    
    # 步骤 2: 图像关键帧生成
    keyframe_path = os.path.join(output_dir, "keyframe.png")
    generate_image(prompt=image_prompt, output_path=keyframe_path)
    
    # 步骤 3: 图像生视频
    video_path = os.path.join(output_dir, "motion.mp4")
    generate_video(image_path=keyframe_path, prompt=video_motion_prompt, output_path=video_path)
    
    print(f"\n✨ 流水线执行完毕！成果保存在: {output_dir}")
    return {
        "audio": audio_path,
        "keyframe": keyframe_path,
        "video": video_path
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=str, default="在浩瀚的赛博都市中，数据流如同永不熄灭的霓虹瀑布。")
    parser.add_argument("--image_prompt", type=str, default="Futuristic cyberpunk skyscraper city at night, rain reflections, neon lights, highly detailed, photorealistic")
    parser.add_argument("--video_prompt", type=str, default="Slow camera dive through neon towers, rain particles moving")
    parser.add_argument("--name", type=str, default="cyber_demo")
    args = parser.parse_args()
    
    run_storyboard_pipeline(args.script, args.image_prompt, args.video_prompt, args.name)
