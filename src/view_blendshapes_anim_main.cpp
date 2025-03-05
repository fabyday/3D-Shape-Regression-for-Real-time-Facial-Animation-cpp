

#include <igl/opengl/glfw/Viewer.h>

#include "network.h"

#include "blendshapes.h"
#include <Eigen/Core>
#include <Windows.h>

char buffer[PACKET_SIZE] = {};
#include <iostream>

int main() {

	Blendshapes bl;
	load_blendshapes("D:\\lab\\2022\\mycode\\3D-Shape-Regression-for-Real-time-Facial-Animation\\train_dataset\\data1", bl); // load ref mesh
	Eigen::MatrixXd bws;
	std::cout << sizeof(int) << std::endl;
	load_blendshape_weight_animation("D:\\lab\\2022\\mycode\\3D-Shape-Regression-for-Real-time-Facial-Animation\\blendshape_animation", bws);


	Eigen::VectorXd weight_vector(bl.get_blendshape_weight_size());

	Eigen::MatrixXd v;

	igl::opengl::glfw::Viewer viewer;
	// Attach a menu plugin

	// Customize the menu
	double doubleVariable = 0.1f; // Shared between two menus

	// Draw additional windows

	// Plot the mesh
	int i = 0;
	//viewer.data().set_mesh(V, F);
	viewer.callback_init = [&](igl::opengl::glfw::Viewer& viwer) -> bool {
		viewer.data().set_mesh(bl.neutral_pose, bl.F);
		return false;
		};
	viewer.core().animation_max_fps = 240;

	viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer& viwer) -> bool {
		
		weight_vector = bws.row(i++).transpose();
		//weight_vector.setOnes();
		//std::cout << i << std::endl;
		bl.blend(weight_vector, v);
		viwer.data().set_vertices(v);
		
		
		return false;



		};
	viewer.launch();


}