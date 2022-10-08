/*
 * Copyright (C) 2016 Open Source Robotics Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#ifndef GAZEBO_PLUGINS_ANIMATEDOBSTACLEPLUGIN_HH_
#define GAZEBO_PLUGINS_ANIMATEDOBSTACLEPLUGIN_HH_

#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <boost/lexical_cast.hpp>
#include <animated_obstacle_plugin/SetPose.h>
#include <animated_obstacle_plugin/SetVel.h>
#include <animated_obstacle_plugin/GetPose.h>
#include <animated_obstacle_plugin/GetVel.h>
#include <animated_obstacle_plugin/Init.h>
#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <ros/subscribe_options.h>
#include <std_msgs/Float32MultiArray.h>
#include <tf/transform_broadcaster.h>

#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/util/system.hh>

namespace gazebo
{
  class GAZEBO_VISIBLE AnimatedObstaclePlugin : public ModelPlugin
  {
    /// \brief Constructor
    public: 
      AnimatedObstaclePlugin();

      /// \brief Load the actor plugin.
      /// \param[in] _model Pointer to the parent model.
      /// \param[in] _sdf Pointer to the plugin's SDF elements.
      virtual void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf);

      /// \brief Function that is called every update cycle.
      /// \param[in] _info Timing information
    private: 
      void OnUpdate(const common::UpdateInfo &_info);
      void Initialize();
      bool isInitialized();
      bool isStatic();

      /// \brief A node use for ROS transport
      // std::shared_ptr<ros::NodeHandle> rosNode;
      ros::NodeHandlePtr rosNode;

      ros::ServiceServer SetPoseService;

      ros::ServiceServer SetVelService;

      ros::ServiceServer GetPoseService;

      ros::ServiceServer GetVelService;

      ros::ServiceServer InitService;

      /// \brief Helper function to choose a new target location
      void ChooseNewTarget();

      /// \brief Helper function to avoid obstacles. This implements a very
      /// simple vector-field algorithm.
      /// \param[in] _pos Direction vector that should be adjusted according
      /// to nearby obstacles.
      void HandleObstacles(ignition::math::Vector3d &_pos);

      /// \brief Pointer to the parent actor.
      physics::ActorPtr actor;

      /// \brief Pointer to the world, for convenience.
      physics::WorldPtr world;

      /// \brief Velocity of the actor
      ignition::math::Vector3d velocity;

      /// \brief Last position of the actor
      ignition::math::Vector3d last_pos;

      /// \brief List of connections
      std::vector<event::ConnectionPtr> connections;

      /// \brief Time scaling factor. Used to coordinate translational motion
      /// with the actor's walking animation.
      double animationFactor = 1.0;

      /// \brief Time of the last update.
      common::Time lastUpdate;

      /// \brief List of models to ignore. Used for vector field
      std::vector<std::string> ignoreModels;

      /// \brief Custom trajectory info.
      physics::TrajectoryInfoPtr trajectoryInfo;

      // ros::Publisher PosePublisher;
      bool SetPoseCallback(animated_obstacle_plugin::SetPose::Request&,
          animated_obstacle_plugin::SetPose::Response&);

      bool SetVelCallback(animated_obstacle_plugin::SetVel::Request&,
          animated_obstacle_plugin::SetVel::Response&);

      bool GetPoseCallback(animated_obstacle_plugin::GetPose::Request&,
          animated_obstacle_plugin::GetPose::Response&);

      bool GetVelCallback(animated_obstacle_plugin::GetVel::Request&,
          animated_obstacle_plugin::GetVel::Response&);

      bool InitCallback(animated_obstacle_plugin::Init::Request&,
          animated_obstacle_plugin::Init::Response&);

      /// \brief A ROS callbackqueue that helps process messages
      ros::CallbackQueue rosQueue;

      /// \brief A thread the keeps running the rosQueue
      std::thread rosQueueThread;

      void QueueThread();

      double animation_step = 0.005;
      bool is_initialized = false;
      bool is_static = false;
  };
}
#endif
