import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

def draw_robot(ax, title, sensors, actuators):
    # Draw chassis
    chassis = Rectangle((-0.4, -0.1), 0.8, 0.2, edgecolor='black', facecolor='lightgray')
    ax.add_patch(chassis)
    # Draw sensors
    for x, y, label, color in sensors:
        circ = Circle((x, y), 0.05, color=color)
        ax.add_patch(circ)
        ax.text(x, y - 0.12, label, ha='center', va='top', fontsize=6)
    # Draw actuators
    for x, y, label, color in actuators:
        rect = Rectangle((x - 0.05, y - 0.05), 0.1, 0.1, color=color)
        ax.add_patch(rect)
        ax.text(x, y + 0.08, label, ha='center', va='bottom', fontsize=6)
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.4, 0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8)
    ax.set_aspect('equal')

fig, axes = plt.subplots(1, 5, figsize=(15, 3))

# Define robots
robots = [
    ("John Deere\nSee & Spray", 
     [(-0.25, 0.15, "Cameras", 'blue'), (0.25, 0.15, "Cameras", 'blue')],
     [(0, -0.15, "Nozzles", 'orange')]),
    ("Carbon Robotics\nLaserWeeder", 
     [(-0.2, 0.15, "Stereo\nvision", 'blue'), (0.2, 0.15, "Stereo\nvision", 'blue')],
     [(0, -0.15, "Lasers", 'red')]),
    ("Naïo Dino", 
     [(-0.2, 0.15, "LiDAR", 'green'), (0.2, 0.15, "Camera", 'blue')],
     [(0, -0.15, "Blades", 'orange')]),
    ("FarmWise\nVulcan", 
     [(-0.25, 0.15, "Cameras", 'blue'), (0.25, 0.15, "Depth", 'cyan')],
     [(0, -0.15, "Knives", 'orange')]),
    ("EcoRobotix AVO", 
     [(-0.25, 0.15, "Multispectral", 'purple'), (0.25, 0.15, "Camera", 'blue')],
     [(0, -0.15, "Micro\nspray", 'orange')])
]

for ax, (name, sensors, actuators) in zip(axes, robots):
    draw_robot(ax, name, sensors, actuators)

fig.suptitle("Figure X  Commercial robotic weeders and their main sensor–actuator suites", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('M:\Research\Diagrams and Visual Aids\Visual_Aids\commercial_robotic_weeders.png', dpi=300)
plt.show()
