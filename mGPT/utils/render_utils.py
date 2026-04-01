import os
import numpy as np
import trimesh
import pyrender
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm

def render_mesh(img, mesh, face, cam_trans, only_mesh=False):
    """
    Render a mesh onto an image.
    """
    # mesh
    mesh = trimesh.Trimesh(mesh, face)
    rot = trimesh.transformations.rotation_matrix(
        np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')

    camera_center = [1.0*img.shape[1]/2, 1.0*img.shape[0]/2]
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = cam_trans
    camera_pose[:3, :3] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]

    focal_length = 5000
    camera = pyrender.camera.IntrinsicsCamera(
        fx=focal_length, fy=focal_length,
        cx=camera_center[0], cy=camera_center[1])
    scene.add(camera, pose=camera_pose)
 
    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)

    # light
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=5e2)
    light_pose = np.eye(4)
    light_pose = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    scene.add(light, pose=light_pose)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:,:,:3].astype(np.float32)
    valid_mask = (depth > 0)[:,:,None]

    # save to image
    img = rgb * valid_mask + img * (1-valid_mask)
    img = np.array(img, dtype=np.uint8)
    return img

def render_video_from_meshes(verts_list, faces, save_path, fps=20, cam_trans=None, ref_verts_list=None):
    """
    Render a video from a list of meshes (vertices).
    
    Args:
        verts_list: List or array of vertices [T, V, 3]
        faces: Faces array [F, 3]
        save_path: Path to save the .mp4 file
        fps: Frames per second
        cam_trans: Camera translation vector (optional)
        ref_verts_list: Optional list of reference vertices to render side-by-side
    """
    if cam_trans is None:
        cam_trans = np.array([-2.6177440e-03, 0.1, -13], dtype=np.float32)
        
    h, w = 512, 512
    frames = []
    length = len(verts_list)
    
    print(f"Rendering video to {save_path}...")
    for t in tqdm(range(length)):
        # Render Main Mesh
        img_pred = np.zeros((h, w, 3), dtype=np.int8)
        img_pred = render_mesh(img=img_pred, mesh=verts_list[t], face=faces, cam_trans=cam_trans)
        
        if ref_verts_list is not None:
            # Render Reference Mesh
            img_ref = np.zeros((h, w, 3), dtype=np.int8)
            img_ref = render_mesh(img=img_ref, mesh=ref_verts_list[t], face=faces, cam_trans=cam_trans)
            
            # Concatenate side-by-side
            frame = np.concatenate([img_ref, img_pred], axis=1)
        else:
            frame = img_pred
            
        frames.append(frame)
    
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(save_path, fps=fps)
    print(f"Video saved to {save_path}")
