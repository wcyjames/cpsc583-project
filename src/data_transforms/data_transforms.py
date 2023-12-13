from data_transforms.utils import *
from torch_geometric.data import Data

def implicit_feedback_sample_list_to_graph(samples, tf_t=None, label=None, metadata_sample_list=None, crop_radius_meters=7.2):
  ''' a graph of all agents where each agent has a "nearby" feature flag'''


  # filter agents to those w/in crop radius
  # agents shape, after the squeeze, is (timesteps, agents, (x, y, cos, sin))
  agent_positions, agent_base_link_poses = positions_to_robot_frame(samples, '/social_sim/agent_positions', tf_t)

  # zero pad positions up to maxlen to handle timesteps with different numbers of people
  agent_positions, agent_mask = pad_positions(agent_positions)
  # reshape to take distance per agent across all timesteps
  # agent shape is now (timesteps, agents, (x, y, cos, sin))
  nearby_agents, agent_mask, distance_from_robot = filter_nearby(agent_positions, agent_mask, crop_radius_meters)
  nearby_agents = nearby_agents.type(torch.float32)
  agent_mask = agent_mask.type(torch.float32)
  nearby_agent_count = nearby_agents.shape[0]

  # combine agents and followers
  all_nearby = nearby_agents
  timestep_count, all_nearby_count, _ = all_nearby.shape

  # in case there are no nearby agents, return None
  if all_nearby_count <= 0:
    return None

  # reshape to (agents, timesteps * features)
  # all_nearby = all_nearby.reshape((all_nearby_count, -1))

  # node features, shape: (num_nodes, num_node_features, timesteps)
  node_features = np.transpose(all_nearby, (1, 2, 0)) # 4 + 1 + 1 + 36 + 37 + 9 + 6 + 4 + 4 + 9 + 6 = 117
  nearby_features = np.zeros((all_nearby_count, 1, timestep_count))
  distance_features = np.zeros((all_nearby_count, 1, timestep_count))
  follow_features = np.zeros((all_nearby_count, 1, timestep_count))
  eye_features = np.zeros((all_nearby_count, samples[0]['/social_sim/xr/eyes']['eyes'].shape[0], timestep_count))
  lip_features = np.zeros((all_nearby_count, samples[0]['/social_sim/xr/lips']['face'].shape[0], timestep_count))
  head_features = np.zeros((all_nearby_count, 9, timestep_count))
  gaze_features = np.zeros((all_nearby_count, 6, timestep_count))
  follow_mask = find_follower(samples, tf_t, all_nearby_count)
  behavior_features = np.zeros((all_nearby_count, 4, timestep_count))
  robot_pos_features = np.zeros((all_nearby_count, 4, timestep_count))
  goal_to_head_features = np.zeros((all_nearby_count, 9, timestep_count))
  goal_to_gaze_features = np.zeros((all_nearby_count, 6, timestep_count))
  time_elapsed_features = np.zeros((all_nearby_count, 1, timestep_count))
  time_remaining_features = np.zeros((all_nearby_count, 1, timestep_count))

  # Node: 4 + 1 + 1 + 36 + 37 + 9 + 6 + 9 + 6 = 109
  # Edge: 5
  # Global: 4 + 4 = 8
  # 0:94, 102:117
  # 94:102

  for follower_idx in follow_mask:
    if follower_idx != 45:
      return None

  for i in range(timestep_count):
    for j in range(all_nearby_count):
      nearby_features[j][0][i] = agent_mask[i][j]
      follow_features[j][0][i] = 1 if follow_mask[i] == j else 0
      eye_features[j,:,i] = samples[i]['/social_sim/xr/eyes']['eyes'] if follow_mask[i] == j else np.zeros_like(samples[i]['/social_sim/xr/eyes']['eyes'])
      lip_features[j,:,i] = samples[i]['/social_sim/xr/lips']['face'] if follow_mask[i] == j else np.zeros_like(samples[i]['/social_sim/xr/lips']['face'])
      
      head_pose_stamped = pose_data_to_pose_stamped(samples[i]['player_head'], rospy.Time(nsecs=int(samples[i]['ts'])), 'map')
      head_np_pose = rebase_pose(head_pose_stamped, tf_t, get3D=True)['featurized_point'].squeeze()
      head_features[j,:,i] = head_np_pose if follow_mask[i] == j else np.zeros_like(head_np_pose)

      gaze_pose_stamped = pose_data_to_pose_stamped(samples[i]['player_gaze'], rospy.Time(nsecs=int(samples[i]['ts'])), 'map')
      gaze_np_pose = rebase_pose(gaze_pose_stamped, tf_t, get3D=True)['featurized_point'].squeeze()[-6:]
      gaze_features[j,:,i] = gaze_np_pose if follow_mask[i] == j else np.zeros_like(gaze_np_pose)

      behavior_features[j,:,i] = samples[i]['/behavior_selector/behavior_name']['behavior_name']

      goal_pose_stamped = pose_data_to_pose_stamped(samples[i]['/move_base_simple/goal']['position'], rospy.Time(nsecs=int(samples[i]['ts'])), 'map')
      goal_np_pose = rebase_pose(goal_pose_stamped, tf_t)['featurized_point'].squeeze()
      robot_pos_features[j,:,i] = goal_np_pose

      goal_to_head_position, goal_to_head_quad = transformation_between_poses(goal_pose_stamped.pose, head_pose_stamped.pose)
      goal_to_head_np_pose = pose_to_numpy_3D(pose_data_to_pose_stamped({'position': goal_to_head_position, 'orientation': goal_to_head_quad}, rospy.Time(nsecs=int(samples[i]['ts'])), 'map').pose).squeeze()
      goal_to_head_features[j,:,i] = goal_to_head_np_pose if follow_mask[i] == j else np.zeros_like(goal_to_head_np_pose)

      goal_to_gaze_position, goal_to_gaze_quad = transformation_between_poses(goal_pose_stamped.pose, gaze_pose_stamped.pose)
      goal_to_gaze_np_pose = pose_to_numpy_3D(pose_data_to_pose_stamped({'position': goal_to_gaze_position, 'orientation': goal_to_gaze_quad}, rospy.Time(nsecs=int(samples[i]['ts'])), 'map').pose).squeeze()[-6:]
      goal_to_gaze_features[j,:,i] = goal_to_gaze_np_pose if follow_mask[i] == j else np.zeros_like(goal_to_gaze_np_pose)

      time_elapsed_features[j,0,i] = samples[i]['/time_elapsed']['time_elapsed']
      time_remaining_features[j,0,i] = samples[i]['/time_remaining']['time_remaining']

  node_features = np.concatenate((node_features, 
                                  nearby_features, 
                                  follow_features, 
                                  eye_features, 
                                  lip_features, 
                                  head_features, 
                                  gaze_features, 
                                  behavior_features, 
                                  robot_pos_features, 
                                  goal_to_head_features, 
                                  goal_to_gaze_features,
                                  time_elapsed_features,
                                  time_remaining_features), axis=1)
  # convert to torch tensor
  node_features = torch.from_numpy(node_features).float()
  
  # edge indices: graph connectivity, shape: (2, num_edges)
  # fully connected with no self-connections
  edge_indices = torch.zeros((2, all_nearby_count**2 - all_nearby_count), dtype=torch.float32).contiguous()
  # edge attributes: edge feature matrix shape: (num_edges, num_edge_features)
  edge_attributes = torch.zeros((edge_indices.shape[1], 5, timestep_count), dtype=torch.float32)

  # for each agent by every other agent
  i = 0
  for j in range(all_nearby_count):
    for k in range(all_nearby_count):
      if j == k:
        continue
      edge_indices[0][i] = j
      edge_indices[1][i] = k
      # l2 norm between person j and person k
      edge_attributes[i, 0, :] = torch.norm(node_features[j, 0:2, :] - node_features[k, 0:2, :], dim=0)

      for t in range(timestep_count):
        # print([agent_j_pose_stamped.pose.orientation.x, agent_j_pose_stamped.pose.orientation.y, agent_j_pose_stamped.pose.orientation.z, agent_j_pose_stamped.pose.orientation.w])
        j_to_k_np_pose = relative_transform(samples[t]['/social_sim/agent_positions']['positions'][j].squeeze(), samples[t]['/social_sim/agent_positions']['positions'][k].squeeze())
        edge_attributes[i, 1:, t] = torch.from_numpy(j_to_k_np_pose)
      i += 1

  edge_indices = edge_indices.long()

  # Create a transformation pipeline
  # Note: Pre-trained models expect input images normalized in a certain way
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Repeat single channel to get three channels
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  # Load a pre-trained ResNet-18 model
  model = models.resnet18(pretrained=True)

  # Remove the last layer (fully connected layer) to get features
  model = torch.nn.Sequential(*(list(model.children())[:-1]))

  # Set model to evaluation mode
  model.eval()

  # Assuming input image 'img' is a numpy array with shape (seq_line, height, width)
  img_transformed = torch.stack([transform(sample['occupancy_grid']) for sample in samples])
  
  # Move model and data to GPU if available
  if torch.cuda.is_available():
      model = model.cuda()
      img_transformed = img_transformed.cuda()

  # Extract features
  with torch.no_grad():
      # Now the image tensor's shape is (batch_size, channels, width, height)
      # The output will have shape (seq_line, 512, 1, 1) for ResNet-18
      map_features = model(img_transformed)

  # Flatten the output if you want a vector rather than a 1x1 "image" tensor
  map_features = map_features.view(map_features.size(0), -1)  # shape (seq_line, 512)

  # Convert to numpy array if needed
  map_features = map_features.cpu()


  # Build the graph
  # https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data
  # TODO: add the graph
  return Data(
    # x (Tensor, optional) – Node feature matrix with shape [num_nodes, num_node_features]. (default: None)
    x = node_features,
    # edge_index (LongTensor, optional) – Graph connectivity in COO format with shape [2, num_edges]. (default: None)
    edge_index = edge_indices,
    # edge_attr (Tensor, optional) – Edge feature matrix with shape [num_edges, num_edge_features]. (default: None)
    edge_attr = edge_attributes,
    # y (Tensor, optional) – Graph-level or node-level ground-truth labels with arbitrary shape. (default: None)
    y = label,
    # pos (Tensor, optional) – Node position matrix with shape [num_nodes, num_dimensions]. (default: None)
    # **kwargs (optional): Additional attributes, set as properties on the object via setattr
    robot_pos_features = robot_pos_features[0, :, :], 
    behavior_features = behavior_features[0, :, :],
    robot_audio = samples[-1]['/robot_audio']['robot_audio'],
    after_behavior_change = samples[-1]['/after_behavior_change']['after_behavior_change'],
    cropped_occupancy_grids = map_features
  )


def get_controller_sample_list_to_trajectory(trajectory_timesteps=5, crop_radius_meters=7.2):

  # closed function w/ trajectory_timesteps and crop_radius_meters
  def controller_sample_list_to_trajectory(sample_list_with_tf_t):
    # pick the current timestep, past and future
    history_sample_idx = -1*trajectory_timesteps
    future_sample_idx = -1*trajectory_timesteps
    global_plan_sample_idx = -1*trajectory_timesteps-1

    sample_list = sample_list_with_tf_t['samples']
    tf_t = sample_list_with_tf_t['tf_t']

    samples = list(load_sample_from_list(sample_list))

      # raise RuntimeError(f"""
      # It looks like extraction was run with --sample-timesteps {len(samples)} and --trajectory-timesteps {trajectory_timesteps}.
      # However, --trajectory_timesteps is too large, or --sample-timesteps is too small.

      # > Note: --sample-timesteps must be greater than --trajectory-timesteps
      # >   --sample-timesteps defines how many samples are initially aggregated into a single sample
      # >   then, of those sample-timesteps, --trajectory-timesteps defines how many are used for the future path (the Y value) and the rest and the data over
      # >   which a prediction is made (the x value)
      # """)

    # filter agents to those w/in crop radius
    # agents shape, after the squeeze, is (timesteps, agents, (x, y, cos, sin))
    agents = torch.tensor([s['/social_sim/agent_positions']['positions'] for s in samples[:history_sample_idx]]).squeeze(2)
    # reshape to take distance per agent across all timesteps
    # agent shape is now (agents, timesteps, (x, y, cos, sin))
    nearby_agents = filter_nearby(agents, crop_radius_meters).type(torch.float32)
    nearby_agent_count = nearby_agents.shape[0]
    # filter followers to those w/in crop radius
    followers = torch.tensor([s['/social_sim/follow_agents']['positions'] for s in samples[:history_sample_idx]])
    if followers.numel() != 0:
      followers = followers.squeeze(2)
      nearby_followers = filter_nearby(followers, crop_radius_meters).type(torch.float32)
    else:
      nearby_followers = followers.type(torch.float32)

    # combine agents and followers
    all_nearby = torch.cat((nearby_agents, nearby_followers), axis=0)
    all_nearby_count, timestep_count, _ = all_nearby.shape

    # in case there are no nearby agents, return None
    if all_nearby_count <= 0:
      return None

    # reshape to (agents, timesteps * features)
    all_nearby = all_nearby.reshape((all_nearby_count, -1))
    # set follower feature (default to non-follower)
    follower_feature = torch.zeros((all_nearby_count, 1))
    # set the follower agent feature values to 1
    follower_feature[nearby_agent_count:] = 1

    # node features, shape: (num_nodes, num_node_features)
    node_features = torch.cat((all_nearby, follower_feature), axis=1)
    # edge indices: graph connectivity, shape: (2, num_edges)
    # fully connected with no self-connections
    edge_indices = torch.zeros((2, all_nearby_count**2 - all_nearby_count), dtype=torch.float32).contiguous()
    # edge attributes: edge feature matrix shape: (num_edges, num_edge_features)
    edge_attributes = torch.zeros((edge_indices.shape[1], timestep_count), dtype=torch.float32)

    # for each agent by every other agent
    i = 0
    for j in range(all_nearby_count):
      for k in range(all_nearby_count):
        if j == k:
          continue
        edge_indices[0][i] = j
        edge_indices[1][i] = k
        # l2 norm between person j and person k
        edge_attributes[i] = torch.norm(all_nearby[j].reshape((timestep_count, -1))[:, 0:2] - all_nearby[k].reshape((timestep_count, -1))[:, 0:2], dim=1)
        i += 1

    # plan features
    rebase_to_ts = rospy.Time(nsecs=int(sample_list[-1*trajectory_timesteps]['ts']))

    # check that the number of samples extracted match the future and past timesteps
    if len(sample_list) <= trajectory_timesteps:
      raise ValueError(f"Sample list must have more than {trajectory_timesteps} samples")

    # generate the rgb images
    rgb_images = []
    for sample in sample_list[:history_sample_idx]:
      rgb_images.append(pt_to_img_path(sample['pt']))

    # generate the local plan
    local_plan = []
    for sample in samples[future_sample_idx:]:
      local_plan.append(rebase_pose(sample['robot_position'], tf_t, rebase_to_ts)['featurized_point'])

    # get the global plan
    #  these points are already in the base_link frame at the global_plan_sample_idx timestep, which must correspond to the current time (not past or future points)
    global_plan = torch.load(sample_list[global_plan_sample_idx]['pt'])['/move_base/GlobalPlanner/plan']['global_plan']

    # Build the graph
    # https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data
    # TODO: add the graph
    return Data(
      # x (Tensor, optional) – Node feature matrix with shape [num_nodes, num_node_features]. (default: None)
      x = node_features,
      # edge_index (LongTensor, optional) – Graph connectivity in COO format with shape [2, num_edges]. (default: None)
      edge_index = edge_indices,
      # edge_attr (Tensor, optional) – Edge feature matrix with shape [num_edges, num_edge_features]. (default: None)
      edge_attr = edge_attributes,
      # y (Tensor, optional) – Graph-level or node-level ground-truth labels with arbitrary shape. (default: None)
      # pos (Tensor, optional) – Node position matrix with shape [num_nodes, num_dimensions]. (default: None)
      # **kwargs (optional): Additional attributes, set as properties on the object via setattr
      # so we can lookup the images or occupancy grid later
      sample_list=sample_list,
      # images
      rgb_images=rgb_images,
      # local_plan
      local_plan=local_plan,
      # local_plan
      global_plan=global_plan,
      # these are too big to persist 5x times each (b/c of the rolling window)
      #cropped_occupancy_grids = [s['occupancy_grid'] for s in samples],
      #depth_image_pytorch = [s['/center_depth/compressed']['torch_image'] for s in samples],
      #rgb_image_pytorch = [s['/robot_firstperson_rgb/compressed']['torch_image'] for s in samples],
    )

  return controller_sample_list_to_trajectory
