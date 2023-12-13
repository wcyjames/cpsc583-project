from data_transforms.utils import *
from torch_geometric.data import Data

BEHAVIORS = [
    'navstack',
    'spin',
    'wrong_way',
    'wait',
]
LABELS = [
    'competence', 
    'surprise',
    'intention',
]

def implicit_feedback_sample_list_to_flat_nearby_only(samples, tf_t=None, label=None, timesteps=None, hz=10, crop_radius_meters=7.2, out_num_agents=None, image_representation_function=None, follower_idx=45):

  if out_num_agents is None:
    raise ValueError('out_num_agents must be specified')

  if hz != 10:
    samples = downsample(samples, hz)

  samples = samples[-timesteps:]

  # filter agents to those w/in crop radius
  # agents shape, after the squeeze, is (timesteps, agents, (x, y, cos, sin))
  agent_positions, _ = positions_to_robot_frame(samples, '/social_sim/agent_positions', tf_t)

  # reshape to take distance per agent at last timestep
  # agent shape is now ( agents, (x, y, cos, sin))
  # raises a NearbyError on failure
  nearby_agents, nearby_idxes, nearby_agent_mask = filter_nearby_at_most_recent_timestep(agent_positions, crop_radius_meters, out_num_agents, follower_idx)


  # combine agents and followers
  num_timesteps, num_agents_nearby, _ = nearby_agents.shape

  if timesteps > num_timesteps:
    raise ValueError('requested timesteps must be <= all_timesteps')

  # filter to requested number of timesteps
  if timesteps < num_timesteps:
    nearby_agents = nearby_agents[-timesteps:]
    nearby_idxes = nearby_idxes[-timesteps:]
    nearby_agent_mask = nearby_agent_mask[-timesteps:]
  
  if out_num_agents != num_agents_nearby:
    raise ValueError('out_num_agents is not equal to all_nearby_count, this is pry a bug in filter_nearby_at_most_recent_timestep')

  ### features ###

  features = dict(
    physical=dict(
      # (timesteps, 512)
      map_resnet18=image_representation_function(samples),
    ),
    spatial=dict(
      # position of the goal relative to the robot
      # (timesteps, (x,y))
      goal=np.zeros((timesteps, 2)),
      # position of the robot relative to the map
      # (timesteps, (cos,sin))
      robot=np.zeros((timesteps, 2)),
      # position of the follower relative to the robot
      # repeated from the nearby_agents
      # (timesteps, (x,y,cos,sin))
      follower=np.zeros((timesteps, 4)),
      follower_idx = -1,
      # (timesteps, num_agents_nearby, (x,y,cos,sin))
      nearby=nearby_agents,
      gaze=np.zeros((timesteps, 2))
    ),
    facial=dict(
      eye=np.zeros((timesteps, samples[0]['/social_sim/xr/eyes']['eyes'].shape[0])),
      lip=np.zeros((timesteps, samples[0]['/social_sim/xr/lips']['face'].shape[0])),
      head=np.zeros((timesteps, 9)),
    ),
    viz=dict(
      map=[],
      eye_data_validata_bit_mask_left=np.zeros((timesteps, 1)),
      eye_data_validata_bit_mask_right=np.zeros((timesteps, 1)),
      eye_data_validata_bit_mask_combined=np.zeros((timesteps, 1)),
      convergence_distance_validity=np.zeros((timesteps, 1)),
      convergence_distance_mm=np.zeros((timesteps, 1)),
      tracking_improvements=np.zeros((timesteps, max([samples[t]['/social_sim/xr/eyes']['tracking_improvements'].shape[0] for t in range(timesteps)]))),
      tracking_improvements_length=np.zeros((timesteps, 1)),
      gaze_direction_local = np.zeros((timesteps, 3)),
      gaze_direction_global = np.zeros((timesteps, 3)),
      gaze_origin_local = np.zeros((timesteps, 3)),
      gaze_origin_global = np.zeros((timesteps, 3)),
    ),
    other=dict(
      behavior=np.zeros((timesteps, 1)),
      time_elapsed=np.zeros((timesteps, 1)),
      time_remaining=np.zeros((timesteps, 1))
    ),
    y=label
  )

  # get the follower features, if they're nearby
  if follower_idx in nearby_idxes:
    follower_nearby_idx = np.argmax(nearby_idxes == follower_idx)
    features['spatial']['follower'] = nearby_agents[:, follower_nearby_idx, :].squeeze()
    features['spatial']['follower_idx'] = follower_nearby_idx
  else: 
    features['spatial']['follower_idx'] = -1

  for t in range(timesteps):
    # per-timestep features
    features['viz']['map'].append(samples[t]['occupancy_grid'])

    features['other']['behavior'][t] = np.argmax(samples[t]['/behavior_selector/behavior_name']['behavior_name'])

    goal_pose_stamped = pose_data_to_pose_stamped(samples[t]['/move_base_simple/correct_goal']['position'], rospy.Time(nsecs=int(samples[t]['ts'])), 'map')
    goal_in_robot_frame= rebase_pose(goal_pose_stamped, tf_t)['featurized_point'].squeeze()
    features['spatial']['goal'][t] = goal_in_robot_frame[:2]

    features['spatial']['robot'][t] = get_robot_global_pose(tf_t, rospy.Time(nsecs=int(samples[t]['ts']))).squeeze()[-2:]

    gaze_pose_stamped = pose_data_to_pose_stamped(samples[t]['player_gaze'], rospy.Time(nsecs=int(samples[t]['ts'])), 'map')
    gaze_in_robot_frame = rebase_pose(gaze_pose_stamped, tf_t, get3D=True)['featurized_point'].squeeze()
    # take only the z-axis rotation of the gaze in sin,cos representation
    features['spatial']['gaze'][t] = gaze_in_robot_frame[-2:]

    features['other']['time_elapsed'][t] = samples[t]['/time_elapsed']['time_elapsed']
    features['other']['time_remaining'][t] = samples[t]['/time_remaining']['time_remaining']

    features['facial']['eye'][t] = samples[t]['/social_sim/xr/eyes']['eyes']
    features['facial']['lip'][t] = samples[t]['/social_sim/xr/lips']['face']

    features['viz']['eye_data_validata_bit_mask_left'][t] = np.array([samples[t]['/social_sim/xr/eyes']['eye_data_validata_bit_mask_left']])
    features['viz']['eye_data_validata_bit_mask_right'][t] = np.array([samples[t]['/social_sim/xr/eyes']['eye_data_validata_bit_mask_right']])
    features['viz']['eye_data_validata_bit_mask_combined'][t] = np.array([samples[t]['/social_sim/xr/eyes']['eye_data_validata_bit_mask_combined']])
    features['viz']['convergence_distance_validity'][t] = np.array([1 if samples[t]['/social_sim/xr/eyes']['convergence_distance_validity'] else 0])
    features['viz']['convergence_distance_mm'][t] = np.array([samples[t]['/social_sim/xr/eyes']['convergence_distance_mm']])
    features['viz']['tracking_improvements'][t, :samples[t]['/social_sim/xr/eyes']['tracking_improvements'].shape[0]] = samples[t]['/social_sim/xr/eyes']['tracking_improvements']
    features['viz']['tracking_improvements_length'][t] = np.array([samples[t]['/social_sim/xr/eyes']['tracking_improvements'].shape[0]])
    features['viz']['gaze_direction_local'][t] = np.array(samples[t]['/social_sim/xr/gaze_direction_local']['gaze'])
    features['viz']['gaze_direction_global'][t] = np.array(samples[t]['/social_sim/xr/gaze_direction_global']['gaze'])
    features['viz']['gaze_origin_local'][t] = np.array(samples[t]['/social_sim/xr/gaze_origin_local']['gaze'])
    features['viz']['gaze_origin_global'][t] = np.array(samples[t]['/social_sim/xr/gaze_origin_global']['gaze'])

    head_pose_stamped = pose_data_to_pose_stamped(samples[t]['player_head'], rospy.Time(nsecs=int(samples[t]['ts'])), 'map')
    head_in_robot_frame = rebase_pose(head_pose_stamped, tf_t, get3D=True)['featurized_point'].squeeze()
    features['facial']['head'][t] = head_in_robot_frame

  # edge indices: graph connectivity, shape: (2, num_edges)
  # fully connected with no self-connections
  edge_indices = torch.zeros((2, out_num_agents**2 - out_num_agents), dtype=torch.float32).contiguous().long()
  # (num_edges, timesteps, (relative x, relative y, relative cos, relative sin))
  edge_attributes = torch.zeros((edge_indices.shape[1], timesteps, 4), dtype=torch.float32)

  ### edges ###
  edge_idx = 0
  for j in range(out_num_agents):
    for k in range(out_num_agents):
      if j == k:
        continue
      edge_indices[0][edge_idx] = j
      edge_indices[1][edge_idx] = k
      # this is already in relative transform!
      # l2 norm between person j and person k
      #edge_attributes[edge_idx, 0, :] = torch.norm(nearby_agents[:, j, 0:2] - nearby_agents[:, k, 0:2], dim=0)

      for t in range(timesteps):
        # print([agent_j_pose_stamped.pose.orientation.x, agent_j_pose_stamped.pose.orientation.y, agent_j_pose_stamped.pose.orientation.z, agent_j_pose_stamped.pose.orientation.w])
        j_to_k_np_pose = relative_transform(nearby_agents[t, j], nearby_agents[t, k])
        #j_to_k_np_pose = relative_transform(samples[t]['/social_sim/agent_positions']['positions'][j].squeeze(), samples[t]['/social_sim/agent_positions']['positions'][k].squeeze())
        edge_attributes[edge_idx, t] = torch.from_numpy(j_to_k_np_pose)
      edge_idx += 1

  # Build the graph
  # https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data
  features['graph'] = Data(
    x=nearby_agents,
    edge_indices=edge_indices,
    edge_attr=edge_attributes,
    y=label,
  )
  features['after_behavior_change'] = (len(set([np.argmax(samples[t]['/behavior_selector/behavior_name']['behavior_name']) for t in range(timesteps)])) > 1) # np.any(np.array([samples[-1]['/after_behavior_change']['after_behavior_change']]))
  features['final_label'] = np.any(np.array([samples[-1]['/final_label']['final_label']]))
  features['have_label'] = np.any(np.array([samples[-1]['/have_label']['have_label']]))
  features['true_label'] = np.array([samples[-1]['/labels']['competence'], samples[-1]['/labels']['surprise'], samples[-1]['/labels']['intention']])
  for i in range(len(features['true_label'])):
      if features['true_label'][i] == 0:
        features['true_label'][i] = 3

  return features


def get_controller_sample_list_to_implicit_feedback_flat(trajectory_timesteps=5, num_timesteps=80, hz=10, crop_radius_meters=0, num_nodes=None, nearby_only=False, image_representation_function=None):
  # samples = load_sample_from_list(sample_list)
  # use this to compute features
  # samples['/social_sim/xr/eyes']['eyes']
  # raise NotImplementedError("TODO Qiping")

  # closed function w/ trajectory_timesteps and crop_radius_meters
  def controller_sample_list_to_implicit_feedback_graph(sample_list_with_tf_t):
    # pick the current timestep, past and future
    history_sample_idx = -1*trajectory_timesteps

    sample_list = sample_list_with_tf_t['samples']
    tf_t = sample_list_with_tf_t['tf_t']
    label = sample_list_with_tf_t['label']
    label = [label['competence'], label['surprise'], label['intention']]
    for i in range(len(label)):
      if label[i] == 0:
        label[i] = 3

    label = torch.tensor(label, dtype=torch.float32)
    samples = list(load_sample_from_list(sample_list))

    if len(samples) != 120:
      print("WARNING: sample list is not 120 long, returning None")
      return None

    if not samples[:history_sample_idx]:
      raise RuntimeError(f"""
      It looks like extraction was run with --sample-timesteps {len(samples)} and --trajectory-timesteps {trajectory_timesteps}.
      However, --trajectory_timesteps is too large, or --sample-timesteps is too small.

      > Note: --sample-timesteps must be greater than --trajectory-timesteps
      >   --sample-timesteps defines how many samples are initially aggregated into a single sample
      >   then, of those sample-timesteps, --trajectory-timesteps defines how many are used for the future path (the Y value) and the rest and the data over
      >   which a prediction is made (the x value)
      """)

    if nearby_only:
      return implicit_feedback_sample_list_to_flat_nearby_only(
        samples=samples,
        tf_t=tf_t,
        label=label,
        timesteps=num_timesteps,
        hz=hz,
        crop_radius_meters=crop_radius_meters,
        out_num_agents=num_nodes,
        image_representation_function=image_representation_function,
      )
    else:
        raise NotImplementedError("Only nearby is implemented")

  return controller_sample_list_to_implicit_feedback_graph
