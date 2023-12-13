import os
import glob
import torch
import cv2
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

import rospy
from geometry_msgs.msg import PoseStamped, Pose

from tf.transformations import euler_from_quaternion, quaternion_from_euler

from annotators.blip import blip_annotate_image, blip_to_model_format, blip_resize

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

class NearbyError(RuntimeError):
  def __init__(self, message):
    super().__init__(message)

def agents_to_most_recent_location(lst):
  ''' converts a list of agent positions of n timesteps to only the most recent position'''
  return lst[0, :, -1]

def agents_to_all_locations(lst):
  ''' converts a list of agent positions of n timesteps to a flattened array of all positions'''
  return lst[0].reshape(-1, 20)

def load_sample_from_list(sample_list):
  for sample in sample_list:
    yield {**torch.load(sample['pt']), 'pt': sample['pt'], 'ts': sample['ts']}

def crop_resize_img(img, crop_size=512):
  # crop image to a square aspect ratio and resize to 512x512
  h, w = img.shape[:2]
  if h > w:
    img = img[(h-w)//2:(h-w)//2+w, :, :]
  else:
    img = img[:, (w-h)//2:(w-h)//2+h, :]
  img = cv2.resize(img, (crop_size, crop_size))
  return img

def pt_to_img_path(pt):
  res = list(glob.iglob(pt.split('.pt')[0] + '-rgb.jpg')) + list(glob.iglob(pt.split('.pt')[0] + '-rgb.png'))
  return res[0]

def load_cv2_rgb_image_from_sample(sample):
  return cv2.imread(pt_to_img_path(sample['pt']))

def load_image_embedding_from_sample(sample):
  return torch.load(sample['pt'].split('.pt')[0] + '-rgb-image-embed.pt')

def load_or_generate_blip_caption_from_sample(sample, blip_models, perturb=False):
  rgb_fname = pt_to_img_path(sample['pt'])
  fname = sample['pt'].split('.pt')[0] + '-rgb-caption.txt'
  if os.path.exists(fname):
    with open(fname, 'r') as f:
      return f.read()
  else:
    caption = blip_annotate_image(rgb_fname, blip_models, perturb=perturb)['text']
    return caption

def np_image_to_pytorch_format(image):
  # convert to float32
  image = image.astype(np.float32) * (2. / 255.0) - 1
  # Transpose to shape `[batch_size, channels, height, width]`
  image = image[None].transpose(0, 3, 1, 2)
  # convert to torch tensor
  return torch.from_numpy(image)

def filter_nearby(people, agent_mask, crop_radius_meters):
  # people shape must be (timesteps, agents, (x, y, cos, sin))

  # people = people.transpose(0,1)
  # l2 norm from origin to each (x,y) position
  distance_from_robot = people[:,:,0:2].norm(dim=2)

  for i in range(distance_from_robot.shape[0]):
    for j in range(distance_from_robot.shape[1]):
      if distance_from_robot[i,j] > crop_radius_meters:
        agent_mask[i,j] = 0

  return people.type(torch.float32), agent_mask.type(torch.float32), distance_from_robot

def filter_nearby_at_most_recent_timestep(agent_positions, crop_radius_meters, num_agents, follower_idx=45):
  # input people shape must be (timesteps, agents, (x, y, cos, sin))
  num_timesteps, orig_num_agents, num_features = agent_positions.shape

  # all distances from robot at all timesteps
  distances_from_robot = agent_positions[:,:,0:2].norm(dim=2)

  # distance filter at all timesteps
  #distance_filter = agent_positions[distances_from_robot <= crop_radius_meters]

  # find the num_agents closest people at the last timestep
  # people are output in order sorted by the nearest person first in the last timestep
  values, filtered_idxes = torch.topk(distances_from_robot[-1], num_agents, largest=False, sorted=True)

  # select the people within the crop radius
  filtered_idxes_in_range = filtered_idxes[values <= crop_radius_meters]

  if follower_idx not in filtered_idxes_in_range:
    raise NearbyError('Follower not in filtered_idxes_in_range')

  if filtered_idxes_in_range.shape[0] == 0:
    raise NearbyError('No agents within crop radius')

  num_agents_in_range = filtered_idxes_in_range.shape[0]
  # 0-pad
  padding = torch.zeros((num_timesteps, num_agents - num_agents_in_range, num_features))

  # get nearby distances and agents in the last timestep, over all timesteps
  nearby_distances_from_robot = distances_from_robot[:, filtered_idxes_in_range]
  nearby_agents_at_all_timesteps = agent_positions[:, filtered_idxes_in_range]

  # get a mask of all nearby people at the last timestep, over all timesteps
  nearby_at_all_timesteps_mask = (nearby_distances_from_robot <= crop_radius_meters).unsqueeze(2).repeat_interleave(num_features, dim=2)

  # get the mask for the selected people over all timesteps
  selected_part_mask = torch.where(nearby_at_all_timesteps_mask, torch.ones((num_timesteps, num_agents_in_range, num_features)), torch.zeros((num_timesteps, num_agents_in_range, num_features)))

  # agent positions in range over all time
  agent_positions_in_range_over_all_time = torch.where(nearby_at_all_timesteps_mask, nearby_agents_at_all_timesteps, selected_part_mask)

  # padded nearby_agents
  nearby_agent_positions = torch.cat((agent_positions_in_range_over_all_time,padding), dim=1)
  # 1 where the nearby_agent_position is in range, otherwise 0
  mask = torch.cat((selected_part_mask,padding), dim=1)

  return nearby_agent_positions, filtered_idxes_in_range, mask

def batch_to_text_and_image_representation(models, batch, device, trajectory_timesteps=5): 
  history_sample_idx = -1*trajectory_timesteps
  txt_embeds = []
  img_embeds = []
  resnet_embeds = []
  mm_embeds = []
  for samples in batch:
    txte = []
    imge = []
    resnete = []
    mme = []
    for sample in samples[:history_sample_idx]:
      # try to load from cache

      # blip representations
      txt_embedding_path = sample['pt'].split('.pt')[0] + '-rgb-text-embed.pt'
      txt_p_exists = os.path.exists(txt_embedding_path)
      img_embedding_path = sample['pt'].split('.pt')[0] + '-rgb-image-embed.pt'
      img_p_exists = os.path.exists(img_embedding_path)
      mm_embedding_path = sample['pt'].split('.pt')[0] + '-rgb-multimodal-embed.pt'
      mm_p_exists = os.path.exists(mm_embedding_path)

      # resnet representations
      resnet_embedding_path = sample['pt'].split('.pt')[0] + '-rgb-resnet-embed.pt'
      resnet_p_exists = os.path.exists(resnet_embedding_path)

      # all exist, skip all processing
      if not (txt_p_exists and img_p_exists and resnet_p_exists and mm_p_exists):
        img = load_cv2_rgb_image_from_sample(sample)
        # results in shape: batch_size, channels, height, width
        img = np_image_to_pytorch_format(img).to(device)
        blip_img = blip_to_model_format(blip_resize(img))

        # resnet expects 224x224px images
        resnet_size = 224
        img_smallest_edge = min(list(img.shape[-2:]))
        resnet_img = transforms.CenterCrop(img_smallest_edge)(img)
        resnet_img = transforms.Resize(resnet_size, interpolation=InterpolationMode.BICUBIC)(resnet_img)

        caption = load_or_generate_blip_caption_from_sample(sample, models['blip'])
      with torch.no_grad():
        if img_p_exists:
          img_embed = torch.load(img_embedding_path)
        else:
          img_embed = models['blip']['encoder'](blip_img, '', mode='image')[0,0]
        if resnet_p_exists:
          resnet_embed = torch.load(resnet_embedding_path)
        elif 'resnet' in models:
          resnet_embed = models['resnet'](resnet_img)
        if txt_p_exists:
          text_embed = torch.load(txt_embedding_path)
        else:
          text_embed = models['blip']['encoder'](blip_img, caption, mode='text')[0,0]
        if mm_p_exists:
          multimodal_embed = torch.load(mm_embedding_path)
        else:
          multimodal_embed = models['blip']['encoder'](blip_img, caption, mode='multimodal')[0,0]
      imge.append(img_embed)
      if 'resnet' in models:
        resnete.append(resnet_embed)
      txte.append(text_embed)
      mme.append(multimodal_embed)
    img_embeds.append(torch.stack(imge))
    if 'resnet' in models:
      resnet_embeds.append(torch.stack(resnete))
    txt_embeds.append(torch.stack(txte))
    mm_embeds.append(torch.stack(mme))
  if 'resnet' in models:
    resnet_embeds = torch.stack(resnet_embeds)
  else:
    resnet_embeds = None
  return torch.stack(img_embeds), torch.stack(txt_embeds), torch.stack(mm_embeds), resnet_embeds


def pose_to_numpy(pose):
  ''' convert a ROS pose to a numpy array '''
  q = np.array((pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w))
  theta = np.array(euler_from_quaternion(q))[-1]
  return np.array([[pose.position.x, pose.position.y, np.cos(theta), np.sin(theta)]])

def pose_to_numpy_3D(pose):
  ''' convert a ROS pose to a numpy array '''
  q = np.array((pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w))
  theta1, theta2, theta3 = euler_from_quaternion(q)
  return np.array([[pose.position.x, pose.position.y, pose.position.z, np.cos(theta1), np.sin(theta1), np.cos(theta2), np.sin(theta2), np.cos(theta3), np.sin(theta3)]])


def pose_data_to_pose_stamped(pose, stamp, frame_id):
  ''' convert a numpy array into a ROS pose'''
  ps = PoseStamped()
  ps.header.frame_id = frame_id
  ps.header.stamp = stamp
  ps.pose = Pose()
  if type(pose) is np.ndarray:
    pose = pose.squeeze()
    if len(pose.shape) > 1:
      raise ValueError("pose must be a 1D array")
    ps.pose.position.x = pose[0]
    ps.pose.position.y = pose[1]
    ps.pose.position.z = 0
    q = quaternion_from_euler(0, 0, np.arctan2(pose[3], pose[2]))
    ps.pose.orientation.x = q[0]
    ps.pose.orientation.y = q[1]
    ps.pose.orientation.z = q[2]
    ps.pose.orientation.w = q[3]
  if type(pose) is dict:
    ps.pose.position.x = pose['position'][0]
    ps.pose.position.y = pose['position'][1]
    ps.pose.position.z = pose['position'][2]
    ps.pose.orientation.x = pose['orientation'][0]
    ps.pose.orientation.y = pose['orientation'][1]
    ps.pose.orientation.z = pose['orientation'][2]
    ps.pose.orientation.w = pose['orientation'][3]
  return ps


def transformation_between_poses(ps1, ps2):
    rot1 = R.from_quat([ps1.orientation.x, ps1.orientation.y, ps1.orientation.z, ps1.orientation.w])
    rot2 = R.from_quat([ps2.orientation.x, ps2.orientation.y, ps2.orientation.z, ps2.orientation.w])

    # Calculate the relative rotation
    relative_rot = rot1.inv() * rot2
    relative_quat = relative_rot.as_quat()

    # Calculate the relative position
    pos1_to_pos2 = np.array([ps2.position.x, ps2.position.y, ps2.position.z]) - np.array([ps1.position.x, ps1.position.y, ps1.position.z])
    relative_position = rot1.inv().apply(pos1_to_pos2)

    return relative_position, relative_quat

def relative_transform(pose_np1, pose_np2):
    # Calculate the difference in position
    delta_x = pose_np2[0] - pose_np1[0]
    delta_y = pose_np2[1] - pose_np1[1]

    # Calculate the difference in orientation using atan2
    theta1 = math.atan2(pose_np1[3], pose_np1[2])
    theta2 = math.atan2(pose_np2[3], pose_np2[2])
    delta_theta = theta2 - theta1

    # Normalize the angle difference to be in the range [-pi, pi]
    delta_theta = math.atan2(math.sin(delta_theta), math.cos(delta_theta))

    # Compute the relative position in object 1's coordinate frame
    relative_x = delta_x * pose_np1[2] + delta_y * pose_np1[3]
    relative_y = -delta_x * pose_np1[3] + delta_y * pose_np1[2]

    return np.array([relative_x, relative_y, np.cos(delta_theta), np.sin(delta_theta)])


def rebase_pose(pose, tf_t, get3D=False, stamp=None, map_frame='map', robot_frame='base_link'):
  '''
    transform a pose into the base link frame at the start of the sample
  '''
  if type(pose) is not PoseStamped:
    raise ValueError('robot_position must be a PoseStamped')
  if pose.header.frame_id is robot_frame and pose.header.stamp == stamp:
    return pose
  if pose.header.frame_id not in [map_frame, robot_frame]:
    raise ValueError('pose must be in map frame or base_link frame')

  try:
      tf_t.lookupTransform(robot_frame, pose.header.frame_id, pose.header.stamp)
      base_link_pose = tf_t.transformPose(robot_frame, pose)
  except Exception as e:
      # skip this local plan point at this timestep, but don't remove the sample
      print(f"Cannot transform pose: {e}")
      return False

  # position
  transformed_pos = np.array([
      base_link_pose.pose.position.x,
      base_link_pose.pose.position.y,
      base_link_pose.pose.position.z
  ])
  # orientation
  transformed_rotation = [
      base_link_pose.pose.orientation.x,
      base_link_pose.pose.orientation.y,
      base_link_pose.pose.orientation.z,
      base_link_pose.pose.orientation.w
  ]
  theta = euler_from_quaternion(transformed_rotation)[-1]

  return {
    'ts': stamp,
    'position': transformed_pos,
    'orientation': transformed_rotation,
    'featurized_point': pose_to_numpy(base_link_pose.pose) if not get3D else pose_to_numpy_3D(base_link_pose.pose),
    'base_link_pose': base_link_pose.pose,
  }


def positions_to_robot_frame(samples, key, tf_t):
  agent_frame = samples[0][key]['frame']
  agent_np_poses = []
  agent_base_link_poses = []
  for sample in samples:
    positions = []
    base_link_poses = []
    for position in sample[key]['positions']:
      pose_stamped = pose_data_to_pose_stamped(position, rospy.Time(nsecs=int(sample['ts'])), agent_frame)
      rebased_output = rebase_pose(pose_stamped, tf_t)
      np_pose = rebased_output['featurized_point']
      base_link_pose = rebased_output['base_link_pose']
      positions.append(np_pose)
      base_link_poses.append(base_link_pose)
    agent_np_poses.append(positions)
    agent_base_link_poses.append(base_link_poses)
  return torch.tensor(agent_np_poses).squeeze(2), agent_base_link_poses


def get_robot_global_pose(tf_t, stamp):
  (trans, rot) = tf_t.lookupTransform('map', 'base_link', stamp)

  # Get the Euler angles from the quaternion
  (roll, pitch, yaw) = euler_from_quaternion(rot)

  # Convert the orientation into the cos(theta) and sin(theta) form
  cos_theta = math.cos(yaw)
  sin_theta = math.sin(yaw)

  return np.array([[trans[0], trans[1], cos_theta, sin_theta]])

def pad_positions(positions, to_len=None):
  # if to_len is None:
  #   to_len = max([len(p) for p in positions])
  # padded = []
  # mask = []
  # for p in positions:
  #   if len(p) < to_len:
  #     padding = np.zeros((to_len - len(p),) + p[0].shape)
  #     padded.append(np.concatenate((p, padding)))
  #     mask.append(np.zeros_like(padding))
  #   else:
  #     padded.append(np.array(p))
  #     mask.append(np.zeros_like(p))
  # agents = torch.tensor(np.array(padded)).squeeze(2)
  # padding_mask = torch.tensor(np.array(mask)).squeeze(2)
  # return agents, padding_mask

  return positions, torch.tensor(np.ones((positions.shape[0], positions.shape[1])))

def find_follower(samples, tf_t, all_nearby_count):
  '''
    find the closest agent to the ego agent
    returns the index of the closest agent
  '''
  agent_frame = samples[0]['/social_sim/agent_positions']['frame']
  follower_mask = []
  for sample in samples:
    follower_idx = 0
    follower_dist = 100000
    head_pose_stamped = pose_data_to_pose_stamped(sample['player_head'], rospy.Time(nsecs=int(sample['ts'])), agent_frame)
    head_np_pose = pose_to_numpy(head_pose_stamped.pose)
    for idx, position in enumerate(sample['/social_sim/agent_positions']['positions']):
      pose_stamped = pose_data_to_pose_stamped(position, rospy.Time(nsecs=int(sample['ts'])), agent_frame)
      np_pose = pose_to_numpy(pose_stamped.pose)
      dist = np.linalg.norm(head_np_pose.squeeze()[:2] - np_pose.squeeze()[:2])
      if dist < follower_dist:
        follower_dist = dist
        follower_idx = idx
    follower_idx = 45
    follower_mask.append(follower_idx)

  return follower_mask


def downsample(samples, hz):
  # downsample the list of samples from 10hz to the given hz
  # take every 10th sample
  return samples[::(10//hz)]