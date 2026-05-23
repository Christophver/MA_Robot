from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Wir starten einfach nur RViz2 ohne Physik-Ballast
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen'
    )

    return LaunchDescription([
        rviz_node
    ])