import comfy.clip_vision
import comfy.model_management
import comfy.utils
import node_helpers
import nodes
import torch
from comfy.utils import common_upscale
from comfy_api.latest import io


def process_keyframe(keyframe_image, position, width, height, length, image, mask, mask_strength=1.0):
    """
    Helper method to process a keyframe image and update the image tensor and mask.
    Universal approach without conditionals, using VAE's 4-frame grouping.

    Args:
        keyframe_image: The keyframe image to process
        position: Frame position where to place the keyframe (0 for start, length-1 for end, or specific position)
        width: Target width for upscaling
        height: Target height for upscaling
        length: Total video length in frames
        image: The main image tensor to update
        mask: The mask tensor to update
        mask_strength: How much to preserve the keyframe (1.0 = fully preserve, 0.0 = allow generation)

    Returns:
        Updated image and mask tensors
    """
    if keyframe_image is None:
        return image, mask

    """
    Dimensions for image and mask:
        image = torch.ones((length, height, width, 3)) * 0.5
        mask = torch.ones((1, 1, latent.shape[2] * 4, latent.shape[-2], latent.shape[-1]))
    """

    # Convert mask strength to mask value
    # mask_strength: 1.0 = preserve fully -> mask_value: 0.0
    # mask_strength: 0.0 = allow generation -> mask_value: 1.0
    mask_value = 1.0 - mask_strength

    # For simplicity, let's keep the special handling for start/end vs middle
    # This matches the original WanFirstLastFrameToVideo behavior
    is_start = position == 0
    is_end = (position == length - 1) or (position == -1)

    if is_start:
        # Start frame: process from beginning
        processed_image = comfy.utils.common_upscale(
            keyframe_image[:length].movedim(-1, 1), width, height, 'bilinear', 'center'
        ).movedim(1, -1)
        num_frames = processed_image.shape[0]

        # Image placement
        image_start = 0
        image_end = num_frames
        print(f'Placing start frame: {image_start} to {image_end}. {processed_image.shape=} {image.shape=}')
        image[image_start:image_end] = processed_image

        # Mask calculation (with +3 padding as per original)
        mask_start = 0
        mask_end = min(num_frames + 3, mask.shape[2])

    elif is_end:
        # End frame: process from end
        processed_image = comfy.utils.common_upscale(
            keyframe_image[-length:].movedim(-1, 1), width, height, 'bilinear', 'center'
        ).movedim(1, -1)
        num_frames = processed_image.shape[0]

        # Image placement
        image_start = length - num_frames
        image_end = length
        image[image_start:image_end] = processed_image
        print(f'Placing end frame: {image_start} to {image_end}. {processed_image.shape=} {image.shape=}')

        # Mask calculation - the mask indices should match the image indices
        # But mask might be slightly larger due to VAE padding
        # The original WanFirstLastFrameToVideo uses negative indexing: mask[:, :, -num_frames:]
        # This works because it counts from the end of the mask array
        mask_start = mask.shape[2] - num_frames
        mask_end = mask.shape[2]

    else:
        # Middle keyframe: handle VAE 4-frame grouping
        processed_image = comfy.utils.common_upscale(
            keyframe_image[:1].movedim(-1, 1), width, height, 'bilinear', 'center'
        ).movedim(1, -1)

        # Calculate 4-frame group boundaries
        latent_group = position // 4
        image_start = latent_group * 4
        image_end = min((latent_group + 1) * 4, length)

        image_start = position
        image_end = position + 1
        image[image_start - 1 : image_end - 1] = processed_image
        image[image_start:image_end] = processed_image
        image[image_start + 1 : image_end + 1] = processed_image
        print(f'Placing middle frame: {position}. {processed_image.shape=} {image.shape=}')
        """
Placing middle frame: 24. processed_image.shape=torch.Size([1, 480, 320, 3]) image.shape=torch.Size([81, 480, 320, 3])
Placing middle frame: 25. processed_image.shape=torch.Size([1, 480, 320, 3]) image.shape=torch.Size([81, 480, 320, 3])
Placing middle frame: 26. processed_image.shape=torch.Size([1, 480, 320, 3]) image.shape=torch.Size([81, 480, 320, 3])
Placing middle frame: 27. processed_image.shape=torch.Size([1, 480, 320, 3]) image.shape=torch.Size([81, 480, 320, 3])
        """

        # # Fill the entire group with the single frame
        # for i in range(image_start, image_end):
        #     print(f'Placing middle frame: {i}. {processed_image.shape=} {image.shape=}')
        #     image[i] = processed_image[0]

        # Create gradient: center frame has most preservation, surrounding frames have less
        # mask_value: 0.0 = fully preserve, 1.0 = fully generate
        mask_start = position
        mask_end = min(mask_start + 1, mask.shape[2])

        # Surrounding frames: add offset to allow more generation (higher mask value = less preserved)
        # Use smaller offset (0.15) for smoother gradient
        surrounding_mask_value = min(mask_value + 0.15, 1.0)

        # Apply to frame before center
        if mask_start - 1 >= 0:
            mask[:, :, mask_start - 1 : mask_start] = surrounding_mask_value

        # Apply to frame after center
        if mask_start + 1 < mask.shape[2]:
            mask[:, :, mask_start + 1 : mask_end + 1] = surrounding_mask_value

    # Apply mask with converted value
    print(f'Applying mask: {mask_start} to {mask_end}. {mask.shape=} {length=} {mask_strength=} {mask_value=}')
    mask[:, :, mask_start:mask_end] = mask_value

    return image, mask


class WanKeyframes(io.ComfyNode):
    """
    A node that creates video conditioning with keyframe images distributed evenly across the video length.
    Accepts start and end images plus up to 3 additional keyframe images and their optional clip vision outputs.
    Keyframes can be positioned automatically (evenly distributed) or at specific frame positions.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='WanKeyframes',
            category='conditioning/video_models',
            inputs=[
                io.Conditioning.Input('positive'),
                io.Conditioning.Input('negative'),
                io.Vae.Input('vae'),
                io.Int.Input('width', default=832, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input('height', default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input('length', default=81, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Int.Input('batch_size', default=1, min=1, max=4096),
                io.ClipVisionOutput.Input('clip_vision_start_image', optional=True),
                io.ClipVisionOutput.Input('clip_vision_end_image', optional=True),
                io.ClipVisionOutput.Input('clip_vision_keyframe_1', optional=True),
                io.ClipVisionOutput.Input('clip_vision_keyframe_2', optional=True),
                io.ClipVisionOutput.Input('clip_vision_keyframe_3', optional=True),
                io.Image.Input('start_image', optional=True),
                io.Image.Input('end_image', optional=True),
                io.Image.Input('keyframe_image_1', optional=True),
                io.Image.Input('keyframe_image_2', optional=True),
                io.Image.Input('keyframe_image_3', optional=True),
                # Position inputs - -1 for auto, or specific frame number
                io.Int.Input(
                    'keyframe_position_1',
                    default=-1,
                    min=-1,
                    max=nodes.MAX_RESOLUTION,
                    tooltip='Frame position for keyframe 1. Use -1 for auto or enter a specific frame number',
                ),
                io.Int.Input(
                    'keyframe_position_2',
                    default=-1,
                    min=-1,
                    max=nodes.MAX_RESOLUTION,
                    tooltip='Frame position for keyframe 2. Use -1 for auto or enter a specific frame number',
                ),
                io.Int.Input(
                    'keyframe_position_3',
                    default=-1,
                    min=-1,
                    max=nodes.MAX_RESOLUTION,
                    tooltip='Frame position for keyframe 3. Use -1 for auto enter a specific frame number',
                ),
                # Strength inputs - 0.0 to 1.0, where 1.0 means fully preserve the keyframe
                io.Float.Input(
                    'keyframe_strength_start',
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.1,
                    tooltip='Preservation strength for start image. 1.0 = preserve fully, 0.0 = allow full generation',
                ),
                io.Float.Input(
                    'keyframe_strength_end',
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.1,
                    tooltip='Preservation strength for end image. 1.0 = preserve fully, 0.0 = allow full generation',
                ),
                io.Float.Input(
                    'keyframe_strength_1',
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.1,
                    tooltip='Preservation strength for keyframe 1. 1.0 = preserve fully, 0.0 = allow full generation',
                ),
                io.Float.Input(
                    'keyframe_strength_2',
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.1,
                    tooltip='Preservation strength for keyframe 2. 1.0 = preserve fully, 0.0 = allow full generation',
                ),
                io.Float.Input(
                    'keyframe_strength_3',
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.1,
                    tooltip='Preservation strength for keyframe 3. 1.0 = preserve fully, 0.0 = allow full generation',
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name='positive'),
                io.Conditioning.Output(display_name='negative'),
                io.Latent.Output(display_name='latent'),
            ],
        )

    @classmethod
    def execute(
        cls,
        positive,
        negative,
        vae,
        width,
        height,
        length,
        batch_size,
        start_image=None,
        end_image=None,
        keyframe_image_1=None,
        keyframe_image_2=None,
        keyframe_image_3=None,
        keyframe_position_1=-1,
        keyframe_position_2=-1,
        keyframe_position_3=-1,
        keyframe_strength_start=1.0,
        keyframe_strength_end=1.0,
        keyframe_strength_1=1.0,
        keyframe_strength_2=1.0,
        keyframe_strength_3=1.0,
        clip_vision_start_image=None,
        clip_vision_end_image=None,
        clip_vision_keyframe_1=None,
        clip_vision_keyframe_2=None,
        clip_vision_keyframe_3=None,
    ) -> io.NodeOutput:
        # Get VAE properties
        spacial_scale = vae.spacial_compression_encode() if hasattr(vae, 'spacial_compression_encode') else 8
        latent_channels = vae.latent_channels if hasattr(vae, 'latent_channels') else 16

        # Create latent tensor
        latent = torch.zeros(
            [batch_size, latent_channels, ((length - 1) // 4) + 1, height // spacial_scale, width // spacial_scale],
            device=comfy.model_management.intermediate_device(),
        )

        # Create image tensor and mask
        image = torch.ones((length, height, width, 3)) * 0.5
        mask = torch.ones((1, 1, latent.shape[2] * 4, latent.shape[-2], latent.shape[-1]))

        # Process start image (position 0)
        image, mask = process_keyframe(start_image, 0, width, height, length, image, mask, keyframe_strength_start)
        # Collect keyframe data with their positions and strengths
        keyframe_data = []
        if keyframe_image_1 is not None:
            keyframe_data.append((keyframe_image_1, keyframe_position_1, keyframe_strength_1))
        if keyframe_image_2 is not None:
            keyframe_data.append((keyframe_image_2, keyframe_position_2, keyframe_strength_2))
        if keyframe_image_3 is not None:
            keyframe_data.append((keyframe_image_3, keyframe_position_3, keyframe_strength_3))

        # Process keyframes
        if keyframe_data:
            # Separate auto and manual positions
            auto_keyframes = []
            manual_keyframes = []

            for kf_image, kf_position, kf_strength in keyframe_data:
                if kf_position == -1:  # -1 means auto
                    auto_keyframes.append((kf_image, kf_strength))
                else:
                    # Clamp position to valid range (1 to length-2 to avoid overlap with start/end)
                    position = max(1, min(kf_position, length - 2))
                    manual_keyframes.append((kf_image, position, kf_strength))

            # Place manual position keyframes first
            for kf_image, position, kf_strength in manual_keyframes:
                image, mask = process_keyframe(kf_image, position, width, height, length, image, mask, kf_strength)

            # Then distribute auto keyframes evenly
            if auto_keyframes:
                num_auto = len(auto_keyframes)
                segments = num_auto + 1

                # Calculate positions for auto keyframes
                auto_positions = []
                for i in range(num_auto):
                    position = int((i + 1) * length / segments)
                    # Ensure we don't overlap with start or end
                    position = max(1, min(position, length - 2))
                    auto_positions.append(position)

                # Place auto keyframes
                for (kf_image, kf_strength), position in zip(auto_keyframes, auto_positions):
                    image, mask = process_keyframe(kf_image, position, width, height, length, image, mask, kf_strength)

        # Process end image (position length-1)
        image, mask = process_keyframe(end_image, length - 1, width, height, length, image, mask, keyframe_strength_end)

        # Encode the combined image
        concat_latent_image = vae.encode(image[:, :, :, :3])
        mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
        positive = node_helpers.conditioning_set_values(
            positive, {'concat_latent_image': concat_latent_image, 'concat_mask': mask}
        )
        negative = node_helpers.conditioning_set_values(
            negative, {'concat_latent_image': concat_latent_image, 'concat_mask': mask}
        )

        # Process and combine clip vision outputs
        clip_vision_outputs = []

        # Collect all clip vision outputs in order
        for cv_output in [
            clip_vision_start_image,
            clip_vision_keyframe_1,
            clip_vision_keyframe_2,
            clip_vision_keyframe_3,
            clip_vision_end_image,
        ]:
            if cv_output is not None:
                clip_vision_outputs.append(cv_output)

        # Combine clip vision outputs if any exist
        if len(clip_vision_outputs) > 0:
            if len(clip_vision_outputs) == 1:
                clip_vision_output = clip_vision_outputs[0]
            else:
                # Concatenate all penultimate hidden states
                states = torch.cat([cv.penultimate_hidden_states for cv in clip_vision_outputs], dim=-2)
                clip_vision_output = comfy.clip_vision.Output()
                clip_vision_output.penultimate_hidden_states = states

            positive = node_helpers.conditioning_set_values(positive, {'clip_vision_output': clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {'clip_vision_output': clip_vision_output})

        out_latent = {}
        out_latent['samples'] = latent
        return io.NodeOutput(positive, negative, out_latent)


def process_keyframe_vace(
    keyframe_image, position, width, height, length, image, mask, empty_frame_level=0.5, mask_strength=1.0
):
    """
    Helper method to process a keyframe for VACE output (images and masks).
    Similar to process_keyframe but adapted for VACE workflow.

    Args:
        keyframe_image: The keyframe image to process
        position: Frame position where to place the keyframe
        width: Target width
        height: Target height
        length: Total video length in frames
        image: The main image tensor to update (num_frames, H, W, 3)
        mask: The mask tensor to update (num_frames, H, W)
        empty_frame_level: Level for empty frames (0.0-1.0)
        mask_strength: How much to mask this keyframe (1.0 = fully masked/preserved, 0.0 = not masked)

    Returns:
        Updated image and mask tensors
    """
    if keyframe_image is None:
        return image, mask

    # Upscale the keyframe image to target dimensions
    processed_image = common_upscale(keyframe_image[:1].movedim(-1, 1), width, height, 'lanczos', 'disabled').movedim(
        1, -1
    )

    # Convert mask_strength to mask value
    # mask_strength: 1.0 = fully preserve (mask = 0)
    # mask_strength: 0.0 = allow generation (mask = 1)
    mask_value = 1.0 - mask_strength

    # Determine placement based on position
    is_start = position == 0
    is_end = position == length - 1

    if is_start:
        # Start frame(s)
        # Allow multiple frames if keyframe_image has multiple frames
        num_frames = min(keyframe_image.shape[0], length)
        processed_image = common_upscale(
            keyframe_image[:num_frames].movedim(-1, 1), width, height, 'lanczos', 'disabled'
        ).movedim(1, -1)
        image[:num_frames] = processed_image
        mask[:num_frames] = mask_value

    elif is_end:
        # End frame(s)
        processed_image = common_upscale(
            keyframe_image[-length:].movedim(-1, 1), width, height, 'lanczos', 'disabled'
        ).movedim(1, -1)
        num_frames = processed_image.shape[0]

        image_start = length - num_frames - 1
        image_end = length

        print(f'Placing end frame: {image_start} to {image_end}. {processed_image.shape=} {image.shape=}')

        image[image_start:image_end] = processed_image
        mask[image_start:image_end] = mask_value

    else:
        # Middle keyframe - apply to specific position and surrounding frames for smoothness
        position = max(0, min(position, length - 1))

        # Place keyframe at exact position
        image[position] = processed_image[0]
        mask[position] = mask_value

        # Apply with gradient to surrounding frames for smoother transitions
        surrounding_mask_value = min(mask_value + 0.15, 1.0)

        if position - 1 >= 0:
            image[position - 1] = processed_image[0]
            mask[position - 1] = surrounding_mask_value

        if position + 1 < length:
            image[position + 1] = processed_image[0]
            mask[position + 1] = surrounding_mask_value

    return image, mask


class WanKeyframesVACE(io.ComfyNode):
    """
    A node that creates images and masks with keyframes for VACE workflow.
    Similar to WanKeyframes but outputs raw images/masks instead of conditioning.
    Compatible with WanVaceToVideo node.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='WanKeyframesVACE',
            category='conditioning/video_models',
            inputs=[
                io.Int.Input('num_frames', default=81, min=1, max=10000, step=4),
                io.Int.Input('width', default=832, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input('height', default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Float.Input(
                    'empty_frame_level',
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip='White level of empty frames',
                ),
                io.Image.Input('start_image', optional=True),
                io.Image.Input('end_image', optional=True),
                io.Image.Input('keyframe_image_1', optional=True),
                io.Image.Input('keyframe_image_2', optional=True),
                io.Image.Input('keyframe_image_3', optional=True),
                # Position inputs
                io.Int.Input(
                    'keyframe_position_1',
                    default=-1,
                    min=-1,
                    max=10000,
                    tooltip='Frame position for keyframe 1. Use -1 for auto distribution',
                ),
                io.Int.Input(
                    'keyframe_position_2',
                    default=-1,
                    min=-1,
                    max=10000,
                    tooltip='Frame position for keyframe 2. Use -1 for auto distribution',
                ),
                io.Int.Input(
                    'keyframe_position_3',
                    default=-1,
                    min=-1,
                    max=10000,
                    tooltip='Frame position for keyframe 3. Use -1 for auto distribution',
                ),
                # Strength inputs
                io.Float.Input(
                    'keyframe_strength_start',
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.1,
                    tooltip='Preservation strength for start image. 1.0 = preserve fully, 0.0 = allow generation',
                ),
                io.Float.Input(
                    'keyframe_strength_end',
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.1,
                    tooltip='Preservation strength for end image. 1.0 = preserve fully, 0.0 = allow generation',
                ),
                io.Float.Input(
                    'keyframe_strength_1',
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.1,
                    tooltip='Preservation strength for keyframe 1',
                ),
                io.Float.Input(
                    'keyframe_strength_2',
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.1,
                    tooltip='Preservation strength for keyframe 2',
                ),
                io.Float.Input(
                    'keyframe_strength_3',
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.1,
                    tooltip='Preservation strength for keyframe 3',
                ),
            ],
            outputs=[
                io.Image.Output(display_name='images'),
                io.Mask.Output(display_name='masks'),
                io.Int.Output(display_name='num_frames'),
                io.Int.Output(display_name='width'),
                io.Int.Output(display_name='height'),
            ],
        )

    @classmethod
    def execute(
        cls,
        num_frames,
        width,
        height,
        empty_frame_level,
        start_image=None,
        end_image=None,
        keyframe_image_1=None,
        keyframe_image_2=None,
        keyframe_image_3=None,
        keyframe_position_1=-1,
        keyframe_position_2=-1,
        keyframe_position_3=-1,
        keyframe_strength_start=1.0,
        keyframe_strength_end=1.0,
        keyframe_strength_1=1.0,
        keyframe_strength_2=1.0,
        keyframe_strength_3=1.0,
    ) -> io.NodeOutput:
        # Determine dimensions from first available image
        sample_image = start_image if start_image is not None else end_image
        if sample_image is None:
            sample_image = keyframe_image_1 if keyframe_image_1 is not None else keyframe_image_2
        if sample_image is None:
            sample_image = keyframe_image_3

        if sample_image is not None:
            device = sample_image.device
            H, W = sample_image.shape[1], sample_image.shape[2]
        else:
            # No images provided, use target dimensions
            device = torch.device('cpu')
            H, W = height, width

        # Create output batch with empty frames
        out_batch = torch.ones((num_frames, H, W, 3), device=device) * empty_frame_level

        # Create mask tensor (1 = generate, 0 = preserve)
        masks = torch.ones((num_frames, H, W), device=device)

        # Process start image (position 0)
        out_batch, masks = process_keyframe_vace(
            start_image, 0, W, H, num_frames, out_batch, masks, empty_frame_level, keyframe_strength_start
        )

        # Collect keyframe data with positions and strengths
        keyframe_data = []
        if keyframe_image_1 is not None:
            keyframe_data.append((keyframe_image_1, keyframe_position_1, keyframe_strength_1))
        if keyframe_image_2 is not None:
            keyframe_data.append((keyframe_image_2, keyframe_position_2, keyframe_strength_2))
        if keyframe_image_3 is not None:
            keyframe_data.append((keyframe_image_3, keyframe_position_3, keyframe_strength_3))

        # Process keyframes
        if keyframe_data:
            # Separate auto and manual positions
            auto_keyframes = []
            manual_keyframes = []

            for kf_image, kf_position, kf_strength in keyframe_data:
                if kf_position == -1:  # -1 means auto
                    auto_keyframes.append((kf_image, kf_strength))
                else:
                    # Clamp position to valid range (1 to num_frames-2 to avoid overlap with start/end)
                    position = max(1, min(kf_position, num_frames - 2))
                    manual_keyframes.append((kf_image, position, kf_strength))

            # Place manual position keyframes first
            for kf_image, position, kf_strength in manual_keyframes:
                out_batch, masks = process_keyframe_vace(
                    kf_image, position, W, H, num_frames, out_batch, masks, empty_frame_level, kf_strength
                )

            # Then distribute auto keyframes evenly
            if auto_keyframes:
                num_auto = len(auto_keyframes)
                segments = num_auto + 1

                # Calculate positions for auto keyframes
                auto_positions = []
                for i in range(num_auto):
                    position = int((i + 1) * num_frames / segments)
                    # Ensure we don't overlap with start or end
                    position = max(1, min(position, num_frames - 2))
                    auto_positions.append(position)

                # Place auto keyframes
                for (kf_image, kf_strength), position in zip(auto_keyframes, auto_positions):
                    out_batch, masks = process_keyframe_vace(
                        kf_image, position, W, H, num_frames, out_batch, masks, empty_frame_level, kf_strength
                    )

        # Process end image (position num_frames-1)
        out_batch, masks = process_keyframe_vace(
            end_image, num_frames - 1, W, H, num_frames, out_batch, masks, empty_frame_level, keyframe_strength_end
        )

        # Resize images and masks to target dimensions if needed
        if W != width or H != height:
            out_batch = common_upscale(out_batch.movedim(-1, 1), width, height, 'lanczos', 'disabled').movedim(1, -1)
            masks = common_upscale(masks.unsqueeze(1), width, height, 'nearest-exact', 'disabled').squeeze(1)

        return io.NodeOutput(out_batch.cpu().float(), masks.cpu().float(), num_frames, width, height)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {'WanKeyframes': WanKeyframes, 'WanKeyframesVACE': WanKeyframesVACE}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {'WanKeyframes': 'Wan Keyframes', 'WanKeyframesVACE': 'Wan Keyframes VACE'}
