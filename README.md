# WAN Keyframes for ComfyUI

> It helps to control video consistency by allowing additional frames

When you generate longer AI videos, faces drift, objects morph, and the model forgets what it was drawing. This node solves that by letting you add keyframes throughout your video to keep things consistent.

## What Does This Actually Do?

This is a ComfyUI custom node that lets you guide your video generation by planting keyframes exactly where you want them. Think of it like GPS waypoints for your video - you tell the AI "start here, pass through these spots, and end there," and it figures out the smooth path between them.

**TL;DR:** Control video generation with up to 5 keyframe images (start, end, and 3 middle frames), place them automatically or manually, and adjust how strictly the AI follows them.

## Why Use This?

**Generate longer, more consistent videos**
- Instead of stitching together short clips (where faces/objects gradually change), generate one longer video with keyframes
- The model stays on track over more frames when you give it checkpoints

**Keep faces and objects consistent**
- After major changes (character turns around, camera angle shifts), the model can forget what things looked like
- Add a keyframe to "remind" it - like when a face turns back to camera, use a keyframe to lock in the features

**Force the model to respect your frames**
- Sometimes the model ignores your last frame or only partially follows it
- Add multiple keyframes near the end to make it actually draw what you want

**Flexible control**
- Use just first/last frames for simple shots
- Add 1-3 middle keyframes for complex transformations or longer sequences
- Adjust strength per keyframe (sometimes you want a suggestion, sometimes you want strict adherence)
- CLIP Vision support for even better coherence across frames

## Installation

### Via ComfyUI Manager (easiest)
1. Open ComfyUI Manager
2. Search for "WAN Keyframes"
3. Click Install
4. Restart ComfyUI

### Manual Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/cybergrind/comfy-wan-keypoints.git
# Restart ComfyUI
```

## How to Use

1. **Load the node**: Search for "Wan Keyframes" in ComfyUI
2. **Connect your inputs**:
   - Positive/negative conditioning
   - VAE
   - At least one keyframe image (start, end, or middle)
3. **Configure your keyframes**:
   - **Auto mode** (default): Set positions to `-1` and the node evenly distributes them
   - **Manual mode**: Set specific frame numbers (e.g., frame 24, frame 48)
4. **Adjust strength**: Play with the strength values (0.0-1.0) for each keyframe
   - 1.0 = "Follow this image exactly"
   - 0.5 = "Use this as reference but take creative liberties"
   - 0.0 = "This is more of a suggestion really"

## Example Use Cases

**Long-form generation without face drift**
- 81 frame video of a person talking
- Keyframe every 20 frames to keep facial features consistent
- Prevents the gradual "morphing into a different person" problem

**Character with major pose changes**
- Start: Character facing camera
- Keyframe 1: Character turns profile
- Keyframe 2: Character faces away
- Keyframe 3: Character turns back to camera (use this to lock in the face again)
- End: Character facing camera

**Forcing stubborn endings**
- Model keeps ignoring your end frame
- Add keyframe at frame -10 and frame -5 with same/similar image
- Model has no choice but to follow your composition

## Parameters Explained

| Parameter | What It Does |
|-----------|-------------|
| `width`/`height` | Output video dimensions (must be multiples of 16) |
| `length` | Total frames in your video |
| `keyframe_position_X` | Frame number for keyframe X, or -1 for auto-distribution |
| `keyframe_strength_X` | How strictly to follow keyframe X (0.0 = loose, 1.0 = strict) |
| `clip_vision_X` | Optional CLIP vision embeddings for better coherence |

## Technical Notes

- Works with VAE-based video models (tested with Hunyuan Video and similar)
- Handles the VAE's 4-frame grouping automatically (you don't need to think about it)
- Middle keyframes get applied with slight blending to surrounding frames for smoothness
- Length should be a multiple of 4 for best results

## Troubleshooting

**My keyframes aren't showing up**
- Make sure your keyframe images are connected
- Check that positions are within your video length
- Try increasing the strength value

**Output looks jumpy between keyframes**
- Lower the strength values for smoother transitions
- Add more intermediate keyframes
- Make sure your keyframe images are similar enough

**It's too slow**
- This node doesn't add much overhead - slowness is likely from your video model
- Reduce video length or dimensions if needed

## Contributing

Found a bug? Have an idea? Open an issue or PR at [github.com/cybergrind/wan_keypoints](https://github.com/cybergrind/wan_keypoints)

## License

MIT - Go wild, make cool stuff

## Credits

Built by [@cybergrind](https://github.com/cybergrind) for the ComfyUI community

---

*Now go make some videos that actually do what you want them to do.*
