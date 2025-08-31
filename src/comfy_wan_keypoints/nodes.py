import comfy.clip_vision
import comfy.model_management
import comfy.utils
import node_helpers
import nodes
import torch
from comfy_api.latest import io


def process_keyframe(keyframe_image, position, width, height, length, image, mask):
    """
    Helper method to process a keyframe image and update the image tensor and mask.

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
    if keyframe_image is not None:
        if position == 0:
            # Process start image
            processed_image = comfy.utils.common_upscale(
                keyframe_image[:length].movedim(-1, 1), width, height, 'bilinear', 'center'
            ).movedim(1, -1)
            image[: processed_image.shape[0]] = processed_image
            mask[:, :, : processed_image.shape[0] + 3] = 0.0
        elif position == length - 1:
            # Process end image
            processed_image = comfy.utils.common_upscale(
                keyframe_image[-length:].movedim(-1, 1), width, height, 'bilinear', 'center'
            ).movedim(1, -1)
            image[-processed_image.shape[0] :] = processed_image
            mask[:, :, -processed_image.shape[0] :] = 0.0
        else:
            # Process middle keyframe at specific position
            processed_keyframe = comfy.utils.common_upscale(
                keyframe_image[:1].movedim(-1, 1), width, height, 'bilinear', 'center'
            ).movedim(1, -1)
            # Place the keyframe at the specified position
            image[position] = processed_keyframe[0]
            # Mark this frame and a few around it as known
            mask[:, :, position * 4 : (position + 1) * 4] = 0.0

    return image, mask


class WanKeyframes(io.ComfyNode):
    """
    A node that creates video conditioning with keyframe images distributed evenly across the video length.
    Accepts start and end images plus up to 3 additional keyframe images and their optional clip vision outputs.
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

        # Collect and place middle keyframe images evenly
        keyframe_images = [keyframe_image_1, keyframe_image_2, keyframe_image_3]
        keyframe_images = [kf for kf in keyframe_images if kf is not None]

        num_keyframes = len(keyframe_images)
        if num_keyframes > 0:
            # Calculate positions for keyframes
            # If we have n keyframes, divide the length into n+1 segments
            segments = num_keyframes + 1
            for i, kf_image in enumerate(keyframe_images):
                # Position at (i+1)/(segments) of the total length
                position = int((i + 1) * length / segments)
                # Ensure we don't overlap with start or end
                position = max(1, min(position, length - 2))
                # Process and place the keyframe
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
