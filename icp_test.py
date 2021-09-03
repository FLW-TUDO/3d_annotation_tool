#!/usr/bin/python3
import numpy as np
import open3d as o3d
import copy

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    #source_temp.paint_uniform_color([0,1,0])
    o3d.visualization.draw_geometries([source_temp, target])


def main():
    threshold = 0.004
    trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    target = o3d.io.read_point_cloud('/home/gouda/segmentation/clutter_annotation_tool/dataset/scenes_old/00000/assembled_cloud.pcd')
    source_choco = o3d.io.read_point_cloud('/home/gouda/segmentation/clutter_annotation_tool/dataset/objects/choco_box_centered.pcd')
    source_fanta = o3d.io.read_point_cloud('/home/gouda/segmentation/clutter_annotation_tool/dataset/objects/fantacan_segmented.pcd')

    rot_mat = np.array([[0.010759608351454168, 0.8440133164625898, -0.5362142784950248],
                        [0.03877938029993033, 0.5354897015063556, 0.8436509581836448],
                        [0.9991898670881684, -0.02987141132339656, -0.026968653912862864]])
    #rot_mat = np.identity(3)  # noise
    source_choco.rotate(rot_mat, center=source_choco.get_center())
    translation = np.array([0.05999999999999999, -0.085, 0.1])
    #translation += np.array([0.01, 0.01, 0.01])  # noise
    source_choco.translate(translation)

    rot_mat = np.array([[0.886356640870718, 0.3228292193845998, -0.33189335681195126],
                        [-0.2795175423923962, -0.19836248367964487, -0.9394265636883776],
                        [-0.3691095347986704, 0.925436988767992, -0.08558347469422369]])
    #rot_mat = np.identity(3)  # noise#
    source_fanta.rotate(rot_mat, center=source_fanta.get_center())
    translation = np.array([-0.06499999999999999, 0.15500000000000005, -0.02])
    source_fanta.rotate(source_fanta.get_rotation_matrix_from_xyz((np.pi/8, -np.pi/4, np.pi/6)), center=source_fanta.get_center())
    translation += np.array([0.01, 0.01, 0.01])  # noise
    source_fanta.translate(translation)

    o3d.visualization.draw_geometries([source_choco, source_fanta, target])

    print("Initial alignment done!")
    evaluation = o3d.pipelines.registration.evaluate_registration(source_fanta, target, threshold, trans_init)
    print(evaluation)

    print("Apply ICP")
    #target_down = target.voxel_down_sample(voxel_size=0.001)
    #source_down = source_fanta.voxel_down_sample(voxel_size=0.001)
    target_down = target
    source_down = source_fanta
    #reg = o3d.pipelines.registration.registration_icp(source_down, target_down, threshold, trans_init,
    #                                                      o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    #                                                      o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
    #print("Point to Point ICP finished.")
    radius = 0.002
    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=100))
    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=100))
    reg = o3d.pipelines.registration.registration_icp(source_down, target_down, threshold, trans_init,
                                                      o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                                                      o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
    print("Point to Plane ICP finished.")
    draw_registration_result(source_down, target_down, reg.transformation)
    # transform ICP output and feed it to color ICP
    source_down.transform(reg.transformation)
    reg = o3d.pipelines.registration.registration_colored_icp(
        source_down, target_down, radius, trans_init,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=25))
    print("color ICP finished.")

    print("Transformation is:")
    print(reg.transformation)
    draw_registration_result(source_down, target_down, reg.transformation)


if __name__ == '__main__':
        main()
