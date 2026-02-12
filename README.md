conda activate env_isaaclab

[The way to run script]
[training]
~/IsaacLab/isaaclab.sh -p ~/nrs_rl/scripts/skrl/train.py --task Template-Nrs-Rl-v0

[play: after finishing training]
~/IsaacLab/isaaclab.sh -p ~/nrs_rl/scripts/skrl/play.py --task Template-Nrs-Rl-v0

[The way to push into github]
cd ~/nrs_rl/source/nrs_rl/nrs_rl/tasks/manager_based/nrs_rl

git init

git add .

git commit -m "commit message"

(push in gitkraken)
