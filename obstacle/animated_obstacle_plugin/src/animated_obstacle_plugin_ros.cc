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

#include <functional>
#include <ignition/math.hh>
#include <gazebo/physics/physics.hh>
#include <std_msgs/Float32MultiArray.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Pose.h>
#include <cassert>
#include <cmath>
#include <ctime>
#include <chrono>
#include "animated_obstacle_plugin.hh"
#include <ros/console.h>
#include <iostream>
#include <thread>
#include <string>
#include <ros/console.h> //roslogging

#define PI 3.14159265359
using namespace gazebo;
GZ_REGISTER_MODEL_PLUGIN(AnimatedObstaclePlugin)

#define WALKING_ANIMATION "walking"

  /////////////////////////////////////////////////
AnimatedObstaclePlugin::AnimatedObstaclePlugin()
{
}

/////////////////////////////////////////////////
void AnimatedObstaclePlugin::Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
{
  this->actor = boost::dynamic_pointer_cast<physics::Actor>(_model);
  this->world = this->actor->GetWorld();
  this->connections.push_back(event::Events::ConnectWorldUpdateBegin(
        std::bind(&AnimatedObstaclePlugin::OnUpdate, this, std::placeholders::_1)));
  this->velocity = 0.0;

  // Add our own name to models we should ignore when avoiding obstacles.
  this->ignoreModels.push_back(this->actor->GetName());

  // Read in the other obstacles to ignore
  if (_sdf->HasElement("ignore_obstacles"))
  {
    sdf::ElementPtr modelElem =
      _sdf->GetElement("ignore_obstacles")->GetElement("model");
    while (modelElem)
    {
      this->ignoreModels.push_back(modelElem->Get< std::string >());
      modelElem = modelElem->GetNextElement("model");
    }
  }

  // Read in if the actor is static or not
  if (_sdf->HasElement("is_static"))
  {
    std::string is_static_str = _sdf->GetElement("is_static")->Get<std::string>();
    this->is_static = boost::lexical_cast<bool>(is_static_str);
  }

  auto skelAnims = this->actor->SkeletonAnimations();
  if (skelAnims.find(WALKING_ANIMATION) == skelAnims.end())
  {
    gzerr << "Skeleton animation " << WALKING_ANIMATION << " not found.\n";
  }
  else
  {
    // Create custom trajectory
    this->trajectoryInfo.reset(new physics::TrajectoryInfo());
    this->trajectoryInfo->type = WALKING_ANIMATION;
    this->trajectoryInfo->duration = 1.0;
    this->actor->SetCustomTrajectory(this->trajectoryInfo);
  }

  // Initialize ros, if it has not already been initialized.
  if (!ros::isInitialized())
  {
    int argc = 0;
    char **argv = NULL;
    ros::init(argc, argv, "gazebo_client",
        ros::init_options::NoSigintHandler);
  }

  this->rosNode.reset(new ros::NodeHandle("gazebo_client"));

  this->rosNode->setCallbackQueue(&this->rosQueue);

  this->SetPoseService = this->rosNode->advertiseService("/"+this->actor->GetName()+"/SetActorPosition",
      &AnimatedObstaclePlugin::SetPoseCallback, this);

  this->SetVelService = this->rosNode->advertiseService("/"+this->actor->GetName()+"/SetActorVelocity",
      &AnimatedObstaclePlugin::SetVelCallback, this);

  this->GetPoseService = this->rosNode->advertiseService("/"+this->actor->GetName()+"/GetActorPosition",
      &AnimatedObstaclePlugin::GetPoseCallback, this);

  this->GetVelService = this->rosNode->advertiseService("/"+this->actor->GetName()+"/GetActorVelocity",
      &AnimatedObstaclePlugin::GetVelCallback, this);

  this->InitService = this->rosNode->advertiseService("/"+this->actor->GetName()+"/InitializeActor",
      &AnimatedObstaclePlugin::InitCallback, this);

  this->rosQueueThread =
    std::thread(std::bind(&AnimatedObstaclePlugin::QueueThread, this));

  // initialize pose of the actor
  Initialize();
}

void AnimatedObstaclePlugin::Initialize()
{
  // get current pose of the actor
  ignition::math::Pose3d pose = this->actor->WorldPose();
  // compute initial yaw
  ignition::math::Angle yaw = atan2(pose.Pos().Y(), pose.Pos().X()) + PI;
  // update yaw of the agent
  pose.Rot() = ignition::math::Quaterniond(0.5*PI, 0, yaw.Radian() + 0.5*PI);
  this->actor->SetWorldPose(pose, false, false);
  last_pos = pose.Pos();
  // finialize initialization
  this->is_initialized = true;
}

bool AnimatedObstaclePlugin::isInitialized()
{
  if (this->is_initialized) {
    return true;
  } else {
    return false;
  }
}

bool AnimatedObstaclePlugin::isStatic()
{
  if (this->is_static) {
    return true;
  } else {
    return false;
  }
}

/////////////////////////////////////////////////
void AnimatedObstaclePlugin::OnUpdate(const common::UpdateInfo &_info)
{

  // if not initialized, just terminate the function
  if (!isInitialized()) return;
  if (isStatic()) return;

  // Time delta
  double dt = (_info.simTime - this->lastUpdate).Double();

  // State of this actor
  ignition::math::Pose3d pose = this->actor->WorldPose();
  ignition::math::Angle yaw = pose.Rot().Euler().Z() - 0.5*PI;

  // check if any displacement happened
  ignition::math::Vector3d dp = pose.Pos() - last_pos;

  // update yaw
  if (this->velocity.Length() > 0) {
    yaw = atan2(this->velocity.Y(), this->velocity.X());
  } else if (dp.Length() > 0) {
    yaw = atan2(dp.Y(), dp.X());
  }
  pose.Rot() = ignition::math::Quaterniond(0.5*PI, 0, yaw.Radian() + 0.5*PI);
  
  // zero-out unwanted z axis velocity
  this->velocity.Z(0.0);
  // update position
  pose.Pos() = pose.Pos() + this->velocity * dt;

  this->actor->SetWorldPose(pose, false, false);
  last_pos = pose.Pos();

  this->actor->SetScriptTime(this->actor->ScriptTime() +
      (this->animation_step * this->animationFactor));
  this->lastUpdate = _info.simTime;
}

// \Set actor position service callback. Response is empty
bool AnimatedObstaclePlugin::SetPoseCallback(animated_obstacle_plugin::SetPose::Request& req,
    animated_obstacle_plugin::SetPose::Response& res){
  ignition::math::Pose3d pose = this->actor->WorldPose();

  pose.Pos().X(req.x);
  pose.Pos().Y(req.y);
  this->actor->SetWorldPose(pose);

  return true;
}

// \Set actor velocity service callback. Response is empty
bool AnimatedObstaclePlugin::SetVelCallback(animated_obstacle_plugin::SetVel::Request& req,
    animated_obstacle_plugin::SetVel::Response& res){
  this->velocity.X(req.x);
  this->velocity.Y(req.y);

  return true;
}

// \Get actor velocity service callback. Request is empty
bool AnimatedObstaclePlugin::GetPoseCallback(animated_obstacle_plugin::GetPose::Request& req, 
    animated_obstacle_plugin::GetPose::Response& res){
  ignition::math::Pose3d pose = this->actor->WorldPose();
  ignition::math::Vector3d pos = pose.Pos();
  res.x = pos.X();
  res.y = pos.Y();
  res.yaw = pose.Rot().Euler().Z() - 0.5*PI;

  return true;
}

// \Get actor velocity service callback. Request is empty
bool AnimatedObstaclePlugin::GetVelCallback(animated_obstacle_plugin::GetVel::Request& req, 
    animated_obstacle_plugin::GetVel::Response& res){
  res.x = this->velocity.X();
  res.y = this->velocity.Y();

  return true;
}

// \Initialize actor service callback. Request is empty
bool AnimatedObstaclePlugin::InitCallback(animated_obstacle_plugin::Init::Request& req, 
    animated_obstacle_plugin::Init::Response& res){
  this->is_initialized = false;
  Initialize();
  res.is_initialized = isInitialized();
  return true;
}

/// \brief ROS helper function that processes messages
void AnimatedObstaclePlugin::QueueThread()
{
  // It gonna be really slow if you change it to 0
  static const double timeout = 0.01;
  while (this->rosNode->ok())
  {
    this->rosQueue.callAvailable(ros::WallDuration(timeout));
  }
}