import comfy.clip_vision
import comfy.model_management
import comfy.utils
import node_helpers
import nodes
import torch
from comfy_api.latest import io


def process_keyframe(keyframe_image, position, width, height, length, image, mask, mask_strength=0.0):
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

    # For simplicity, let's keep the special handling for start/end vs middle
    # This matches the original WanFirstLastFrameToVideo behavior
    is_start = position == 0
    is_end = position == length - 1

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

        # Mask matches the image group
        mask_start = position
        mask_end = min(mask_start + 1, mask.shape[2])
        mask[:, :, mask_start - 1 : mask_end - 1] = max(mask_strength + 0.3, 1.0)
        mask[:, :, mask_start + 1 : mask_end + 1] = max(mask_strength + 0.3, 1.0)
        mask_strength = mask_strength

    # Apply mask
    print(f'Applying mask: {mask_start} to {mask_end}. {mask.shape=} length={length} {mask_strength=}')
    mask[:, :, mask_start:mask_end] = mask_strength

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
        image, mask = process_keyframe(start_image, 0, width, height, length, image, mask)

        # Collect keyframe data with their positions
        keyframe_data = []
        if keyframe_image_1 is not None:
            keyframe_data.append((keyframe_image_1, keyframe_position_1))
        if keyframe_image_2 is not None:
            keyframe_data.append((keyframe_image_2, keyframe_position_2))
        if keyframe_image_3 is not None:
            keyframe_data.append((keyframe_image_3, keyframe_position_3))

        # Process keyframes
        if keyframe_data:
            # Separate auto and manual positions
            auto_keyframes = []
            manual_keyframes = []

            for kf_image, kf_position in keyframe_data:
                if kf_position == -1:  # -1 means auto
                    auto_keyframes.append(kf_image)
                else:
                    # Clamp position to valid range (1 to length-2 to avoid overlap with start/end)
                    position = max(1, min(kf_position, length - 2))
                    manual_keyframes.append((kf_image, position))

            # Place manual position keyframes first
            for kf_image, position in manual_keyframes:
                image, mask = process_keyframe(kf_image, position, width, height, length, image, mask)

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
                for kf_image, position in zip(auto_keyframes, auto_positions):
                    image, mask = process_keyframe(kf_image, position, width, height, length, image, mask)

        # Process end image (position length-1)
        image, mask = process_keyframe(end_image, length - 1, width, height, length, image, mask)

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


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {'WanKeyframes': WanKeyframes}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {'WanKeyframes': 'Wan Keyframes'}
