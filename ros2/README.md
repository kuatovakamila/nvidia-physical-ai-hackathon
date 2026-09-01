# ROS 2 node

The planner and the phase executor, exposed as a ROS 2 node. No hardware, no
Isaac: the node drives `executor.FakeBackend`, so it runs anywhere ROS does.

```bash
./ros2/run_in_docker.sh          # build + run + exercise it, start to finish
```

## Why this exists

Access was never the obstacle. ROS 2 is free and open source, it runs natively
on Ubuntu, and on any laptop through the official `ros:humble` image or
RoboStack via conda on macOS. What was missing here was a robot to attach it
to — and that turns out not to matter, because the thing worth demonstrating is
the interface, not the actuator.

The credible minimum is exactly this: wrap the executor as a node with
publishers, a subscriber, and a real topic contract, run it against the fake
backend, and have a repo behind it. That is a weekend of work rather than a
project, and it makes "ROS 2" a truthful line rather than an aspiration.

How much it matters is worth being honest about. On most robotics postings it
is the one requirement where a candidate with simulation and hardware work
still shows nothing, and teams treat it as a baseline literacy check, so a
screener may filter on it. It is not what carries an application — real
hardware deployment, Isaac Sim, PyTorch and a paper do more. But being unable
to say anything about it in a screen call is a different problem from not
having used it, and this package removes that.

## Why it was cheap to add

The executor already had one interface and three interchangeable backends:

```
run_goal()  — phases, monitor gates, retries
   ├── FakeBackend   pure Python        (self-tests, and this node)
   ├── SimBackend    Isaac articulation (the eval batches)
   └── TaughtArm     SO-101 over USB    (the physical demo)
```

ROS is a fourth consumer of that seam, not a rewrite of anything. `bridge.py`
holds the logic and imports no ROS at all; `kitchen_node.py` is a thin rclpy
wrapper over it. That split is why the logic can be tested on a machine with no
ROS installed:

```bash
python3 ros2/src/within_reach/within_reach/bridge.py
```

## Topics

| direction | topic | type | meaning |
|---|---|---|---|
| sub | `/within_reach/command` | `std_msgs/String` | what to deliver, e.g. `medicine next to water` |
| pub | `/within_reach/plan` | `std_msgs/String` | one line per goal the planner issues |
| pub | `/within_reach/event` | `std_msgs/String` | monitor verdicts and the final outcome |
| pub | `/joint_states` | `sensor_msgs/JointState` | streamed while the arm executes |

`/joint_states` keeps its conventional name so `rviz2`, `rqt_plot` and
`ros2 bag record` work without configuration.

## By hand

```bash
docker run --rm -it -v "$PWD":/repo -w /repo/ros2 -e WITHIN_REACH_ROOT=/repo ros:humble bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select within_reach
source install/setup.bash
ros2 run within_reach kitchen_node

# in a second shell into the same container
ros2 topic pub --once /within_reach/command std_msgs/String "{data: 'medicine next to water'}"
ros2 topic echo /within_reach/plan
```

## What you should see

The planner decides, unprompted, that the cup is in the way:

```
goal 1: move cup      -> (0.263, 0.026)  [unblock medicine]
goal 2: move medicine -> (0.190, 0.098)  [user]
[plan] medicine blocked by cup -> relocate it
DELIVERED medicine at (0.188, 0.097)
```

Nowhere is "move the cup first" written down. It falls out of the goal stack —
the same code that scores 40% against 100% over twenty episodes in simulation,
and that drives the physical SO-101 through taught poses.
