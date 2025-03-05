#pragma once 
#pragma comment(lib,"Ws2_32.lib")
#include <WinSock2.h>
#include <iostream>
#include <Eigen/Core>

#define PORT 5090
#define PACKET_SIZE 120
#define LOCAL_SERVER_IP "127.0.0.1"

struct networkCtx {
	WSADATA wasData;
	SOCKET listen_o_con_sock; // if client program it'll become sender, if server it is listen socket
	SOCKET src_o_dest_sock; // if client do nothing, and server will use to direction of stream. 
};


inline int  create_client_socket(networkCtx& ctx) {

	WSAStartup(MAKEWORD(2, 2), &(ctx.wasData));

	
	ctx.listen_o_con_sock= socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);

	SOCKADDR_IN tAddr = {};
	tAddr.sin_family = AF_INET;
	tAddr.sin_port = htons(PORT);
	tAddr.sin_addr.s_addr = inet_addr(LOCAL_SERVER_IP);
	int time_ = 0;
	while (true) {
		int res = connect(ctx.listen_o_con_sock, (SOCKADDR*)&tAddr, sizeof(tAddr));
		if (res == 0) 
			break;
		else {
			if (time_ % 1000 == 0) {
				std::cout << "wait to connect server " << time_ % 1000 << "sec..." << std::endl;
			}
		}
		Sleep(1000);
	}
	std::cout << "connected!" << std::endl;
	return 1;
}

inline int  create_server_socket(networkCtx& ctx) {

	int iResult = WSAStartup(MAKEWORD(2, 2), &(ctx.wasData) );
	if (iResult != NO_ERROR) {
		wprintf(L"WSAStartup() failed with error: %d\n", iResult);
		return 0;
	}
	ctx.listen_o_con_sock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
	if (ctx.listen_o_con_sock == INVALID_SOCKET) {
		wprintf(L"socket function failed with error: %ld\n", WSAGetLastError());
		WSACleanup();
		return 0;
	}
	u_long mode = 0;

	

	SOCKADDR_IN tListenAddr = {};
	tListenAddr.sin_family = AF_INET;
	tListenAddr.sin_port = htons(PORT);
	tListenAddr.sin_addr.s_addr = htonl(INADDR_ANY);
	bind(ctx.listen_o_con_sock, (SOCKADDR*)&tListenAddr, sizeof(tListenAddr));
	int lres = listen(ctx.listen_o_con_sock, SOMAXCONN);


	return 1;
}



inline void  accept(networkCtx& ctx) {
	SOCKADDR_IN tClntAddr = {};
	int iClntSize = sizeof(tClntAddr);
	SOCKET hClient = accept(ctx.listen_o_con_sock, (SOCKADDR*)&tClntAddr, &iClntSize);
	std::cout << "accepted : " << hClient << std::endl;
	ctx.src_o_dest_sock = hClient;
	u_long mode = 1;

	int IRESULT = ioctlsocket(ctx.src_o_dest_sock, FIONBIO, &mode);
	if (IRESULT != NO_ERROR) {
		std::cout << "ioctlsocket fialed with error" << IRESULT << std::endl;
	}

}

inline int recv_blendshapes(Eigen::VectorXd& out, networkCtx& ctx) {
	int ret = recv(ctx.src_o_dest_sock, reinterpret_cast<char*>(out.data()), out.size() * sizeof(double), 0);
	if (ret == SOCKET_ERROR) {
		/*if (WSAGetLastError() == WSAEWOULDBLOCK)
			std::cerr << WSAGetLastError() << std::endl;
		return 0;*/
		return 0;
	}
	else if(ret == 0){
		std::cerr << "connection closed" << std::endl;
		exit(0);
	}
	return 1;
}
inline int send_blendshapes(Eigen::VectorXd& in, networkCtx& ctx){
		
	send(ctx.listen_o_con_sock, reinterpret_cast<char*>(in.data()), sizeof(double) * in.size(), 0);
	return 1;
}

inline void cleanup(networkCtx& ctx) {
	closesocket(ctx.src_o_dest_sock);
	closesocket(ctx.listen_o_con_sock);
	WSACleanup();
}



struct CProcessInfo {
	STARTUPINFO si;
	PROCESS_INFORMATION pi;
};


inline int create_child_viewer(CProcessInfo& ctx) {
	STARTUPINFO si;
	PROCESS_INFORMATION pi;

	ZeroMemory(&ctx.si, sizeof(ctx.si));
	si.cb = sizeof(si);
	ZeroMemory(&ctx.pi, sizeof(ctx.pi));

	if (!CreateProcessA(NULL, "./viewer.exe",
		NULL, NULL, FALSE,
#if _DEBUG
		CREATE_NEW_CONSOLE | DEBUG_PROCESS,
#else
		CREATE_NEW_CONSOLE,
#endif
		NULL, NULL, &ctx.si, &ctx.pi )) {
		std::cout << "create process failed" << std::endl;
		return 0;
	}


	//OpenProcess(PROCESS_ALL_ACCESS, true, )

	return 1;

}


inline int close_child_handle(CProcessInfo& ctx) {
	CloseHandle(ctx.pi.hProcess);
	CloseHandle(ctx.pi.hThread);

	return 1;
}


