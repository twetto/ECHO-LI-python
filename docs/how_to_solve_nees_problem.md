You asked:
----------

你說了

So, to sum up:

1\. Initialize the point at detection, with its id, reference x, y, pose

2\. Track the point, and propagate using VIO pose covariance

3\. When parallax is above certain threshold, take measurement by triangulating the point, and update the reference x, y, pose.

---

Gemini Replied:
---------------

That is the exact architectural blueprint.

To turn that summary into a flawless implementation checklist, I would just add one tiny mathematical safety check to your third step:

1.  **Initialize (Spawn):** At detection, initialize the feature with its ID, and save the current frame's pixel coordinates ( $x,y$ ) and camera pose as the anchor reference.
    
2.  **Track & Coast (Predict):** Track the point frame-to-frame using Lucas-Kanade. Predict the state forward by propagating the depth uncertainty using the EqF's spatial velocity covariance ( $P_{UU}$ ) to account for uncertain rotations and translations. **Do not update the mixture.**
    
3.  **Gate, Update, & Re-Anchor:**
    
    *   _Safety Check:_ Ensure the physical baseline from the reference pose is > 1cm (to prevent division-by-zero during pure camera rotation).
        
    *   _Parallax Gate:_ Calculate the derotated optical flow magnitude. If it is below your threshold (e.g., 3.0 pixels), keep coasting.
        
    *   _Update:_ If the parallax exceeds the threshold, triangulate to get  $z_{obs}$ , compute the observation variance  $\tau ^{2}$ , and update the Vogiatzis mixture.
        
    *   _Re-Anchor:_ Immediately overwrite the reference pixel coordinates and pose with the _current_ frame's data to cleanly reset the Markov chain.
        

By following this exact logic, you are building the holy grail of VIO depth filters: you get the blazing  $O\left(N\right)$  computational speed of optical flow, the geometric robustness of keyframe-based systems like SVO and REMODE, and the statistical consistency (perfect NEES) of a proper Equivariant Filter.

