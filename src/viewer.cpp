

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

	networkCtx server_ctx;
	

	create_server_socket(server_ctx);
	accept(server_ctx);
	std::cout << "accepted" << std::endl;
	



	Eigen::VectorXd weight_vector(bl.get_blendshape_weight_size());

	Eigen::MatrixXd v;

	igl::opengl::glfw::Viewer viewer;
	// Attach a menu plugin

	// Customize the menu
	double doubleVariable = 0.1f; // Shared between two menus

	// Draw additional windows

	// Plot the mesh
	//viewer.data().set_mesh(V, F);
	viewer.callback_init = [&](igl::opengl::glfw::Viewer& viwer) -> bool{
		viewer.data().set_mesh(bl.neutral_pose, bl.F);
		return false;
	};
	viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer& viwer) -> bool{

		if (recv_blendshapes(weight_vector, server_ctx)) {
			bl.blend(weight_vector, v);
			viwer.data().set_vertices(v);
		}
		return false;
	};
	viewer.launch();
	
	cleanup(server_ctx);

}