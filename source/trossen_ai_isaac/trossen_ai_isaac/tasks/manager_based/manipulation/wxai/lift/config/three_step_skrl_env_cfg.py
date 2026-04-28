from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from .. import mdp

from .joint_pos_env_cfg import WXAICubeLiftEnvCfg


@configclass
class WXAICubeLiftSkrlObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        pose_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "object_pose"}
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class TeacherCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_pos_in_ee = ObsTerm(func=mdp.object_position_in_ee_frame)
        
        pose_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "object_pose"}
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class StudentCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        # TODO: RGB termを追加（camera sensorを導入後にObsTermを追加）

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    teacher: TeacherCfg = TeacherCfg()
    student: StudentCfg = StudentCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # 1.1, 15.0, 16.0, 5.0はそれぞれreaching, lifting, coarse/fine object_goal_trackingの重み。rewardの値自体は、stdなどのtermのパラメータによって変わるため、重みはあくまで相対的なもの。

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=10.0)

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=0.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=0.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=0.0,
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_[0-5]", "left_carriage_joint"])},
    )

    # pre_grasp = RewTerm(func=mdp.pre_grasp, params={"distance_std": 0.10, "rotation_std": 0.1}, weight=10.0)

    object_inside_gripper_binary = RewTerm(func=mdp.object_inside_gripper_binary, weight=100.0)


### カリキュラム学習用のconfig
### 指定ステップを超えたタイミングで報酬の重みをここで設定したものに上書きする
@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000},
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000},
    )

    # lifting_object = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "lifting_object", "weight": 10.0, "num_steps": 10000},
    # )

    # reaching_object = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "reaching_object", "weight": 1.0, "num_steps": 10000},
    # )

    # pre_grasp = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "pre_grasp", "weight": 20.0, "num_steps": 10000},
    # )




@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    ### 時間切れでエピソード終了
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    ### 物体を机から落としたらエピソード終了
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )

@configclass
class WXAICubeLiftSkrlEnvCfg(WXAICubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Set actions for the WXAI robot
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_[0-5]"],
            scale=0.5,
            use_default_offset=True,
        )
        # self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=["left_carriage_joint"],
        #     open_command_expr={"left_carriage_joint": 0.044},
        #     close_command_expr={"left_carriage_joint": 0.0},
        # )

        self.actions.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["left_carriage_joint"],
            scale=0.5,
            use_default_offset=True
        )

        self.observations = WXAICubeLiftSkrlObservationsCfg()
        self.rewards = RewardsCfg()
        self.terminations = TerminationsCfg()
        self.curriculum = CurriculumCfg()


@configclass
class WXAICubeLiftSkrlEnvCfg_PLAY(WXAICubeLiftSkrlEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.observations.teacher.enable_corruption = False
        self.observations.student.enable_corruption = False