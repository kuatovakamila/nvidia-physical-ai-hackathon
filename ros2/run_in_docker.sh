#!/usr/bin/env bash
# Build and exercise the node in the official ROS 2 image.
#
# There is no ROS on this laptop and no /opt/ros, which is the usual reason a
# "ROS 2" line on a CV has nothing runnable behind it. The official image
# removes that excuse: one command builds the package, starts the node,
# publishes a command to it, and prints what came back on the topics.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

docker run --rm -v "$REPO":/repo -w /repo/ros2 -e WITHIN_REACH_ROOT=/repo \
  ros:humble bash -lc '
set -e
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select within_reach 2>&1 | tail -3
source install/setup.bash

ros2 run within_reach kitchen_node > /tmp/node.log 2>&1 &
NODE=$!
sleep 4

echo "--- topics ---"
ros2 topic list | grep -E "within_reach|joint_states"

ros2 topic echo /within_reach/plan  --once > /tmp/plan.log 2>&1 &
ros2 topic echo /joint_states       --once > /tmp/joints.log 2>&1 &
sleep 1

ros2 topic pub --once /within_reach/command std_msgs/String "{data: \"medicine next to water\"}" >/dev/null
sleep 6

echo "--- node log ---"
grep -E "command:|goal |\[plan\]|DELIVERED|done:" /tmp/node.log | head -20
echo "--- first /within_reach/plan message ---"
cat /tmp/plan.log
echo "--- first /joint_states message (truncated) ---"
head -12 /tmp/joints.log
kill $NODE 2>/dev/null || true
'
