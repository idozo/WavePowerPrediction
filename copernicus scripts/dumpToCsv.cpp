
#include <iostream>
#include <fstream>
#include <netcdf>
#include <cmath>
using namespace std;
using namespace netCDF;
using namespace netCDF::exceptions;
static const int NTIME = 24;
static const int NLAT = 1;//380
static const int NLON = 1;//1307


#define INVALID_VAL -1000000
static const int NC_ERR = 2;

int main(int argc, const char**argv)
{

// calc data 
	float lats[NLAT], lons[NLON];
	double times[NTIME];
	short sig_hight_in[1][NLAT][NLON];
	short wave_period_in[1][NLAT][NLON];
	short VTM02_in[1][NLAT][NLON];
	short VMDR_in[1][NLAT][NLON];
	short VSDX_in[1][NLAT][NLON];
	short VSDY_in[1][NLAT][NLON];
	short VHM0_WW_in[1][NLAT][NLON];
	short VTM01_WW_in[1][NLAT][NLON];
	short VMDR_WW_in[1][NLAT][NLON];
	short VHM0_SW1_in[1][NLAT][NLON];
	short VTM01_SW1_in[1][NLAT][NLON];
	short VMDR_SW1_in[1][NLAT][NLON];
	short VHM0_SW2_in[1][NLAT][NLON];
	short VTM01_SW2_in[1][NLAT][NLON];
	short VMDR_SW2_in[1][NLAT][NLON];
	short VPED_in[1][NLAT][NLON];
	short VTPK_in[1][NLAT][NLON];
	
 


	float *** wave_power = new float **[NTIME];
	for (int i = 0; i < NTIME; i++) {
		wave_power[i] = new float * [NLAT];
		for (int j = 0; j < NLAT; j++) {
			wave_power[i][j] = new float[NLON];
		}
	}
	std::string year;

	try{
		year = std::string(argv[3]);
		std::string filenameNc = "./data/";
		filenameNc += argv[1];//day
		filenameNc += argv[2];//month
		filenameNc += year;
		filenameNc += ".nc";
		
		bool first_time = false;
		ifstream ifile;
		ifile.open("database_" + year +".csv");
		if(!ifile) {
			first_time = true;
		} else{
			ifile.close();
		}

		NcFile inputFile(filenameNc, NcFile::read);
		std::ofstream myfile;
		myfile.open ("database_" + year +".csv", std::ofstream::out | std::ofstream::app);//appends
		if(myfile.is_open() == 0)
			return 1;
		if (first_time){
			myfile << "time,hour,day,month,VHM0,VTM10,VTM02,VMDR,VSDX,VSDY,VHM0_WW,VTM01_WW,VMDR_WW,VHM0_SW1,VTM01_SW1,VMDR_SW1,VHM0_SW2,VTM01_SW2,VMDR_SW2,VPED,VTPK,wavepower" << std::endl; //use only at first run 
		}
		cout << "writing from file: " << filenameNc << endl;

    // get vars
		NcVar timeVar, sigHightVar, longitudeVar, latitudeVar, wavePeriodVar, VTM02Var,  VMDRVar,  VSDXVar,  VSDYVar,  VHM0_WWVar,  VTM01_WWVar,  VMDR_WWVar,  VHM0_SW1Var,  VTM01_SW1Var,  VMDR_SW1Var,  VHM0_SW2Var,  VTM01_SW2Var,  VMDR_SW2Var,  VPEDVar,  VTPKVar;

		timeVar = inputFile.getVar("time");
		if(timeVar.isNull())
			return NC_ERR;
		timeVar.getVar(times);


		longitudeVar = inputFile.getVar("longitude");
		if(longitudeVar.isNull())
			return NC_ERR;

		longitudeVar.getVar(lons);


		latitudeVar = inputFile.getVar("latitude");
		if(latitudeVar.isNull())
			return NC_ERR;
		latitudeVar.getVar(lats);


		sigHightVar = inputFile.getVar("VHM0");

		if(sigHightVar.isNull())
			return NC_ERR;

		wavePeriodVar = inputFile.getVar("VTM10");

		if(wavePeriodVar.isNull())
			return NC_ERR;

		VTM02Var = inputFile.getVar("VTM02");

		if(VTM02Var.isNull())
			return NC_ERR;
		VMDRVar = inputFile.getVar("VMDR");

		if(VMDRVar.isNull())
			return NC_ERR;
		VSDXVar = inputFile.getVar("VSDX");

		if(VSDXVar.isNull())
			return NC_ERR;
		VSDYVar = inputFile.getVar("VSDY");

		if(VSDYVar.isNull())
			return NC_ERR;
		VHM0_WWVar = inputFile.getVar("VHM0_WW");

		if(VHM0_WWVar.isNull())
			return NC_ERR;
		VTM01_WWVar = inputFile.getVar("VTM01_WW");

		if(VTM01_WWVar.isNull())
			return NC_ERR;
		VMDR_WWVar = inputFile.getVar("VMDR_WW");

		if(VMDR_WWVar.isNull())
			return NC_ERR;
		VHM0_SW1Var = inputFile.getVar("VHM0_SW1");

		if(VHM0_SW1Var.isNull())
			return NC_ERR;
		VTM01_SW1Var = inputFile.getVar("VTM01_SW1");

		if(VTM01_SW1Var.isNull())
			return NC_ERR;
		VMDR_SW1Var = inputFile.getVar("VMDR_SW1");

		if(VMDR_SW1Var.isNull())
			return NC_ERR;
		VHM0_SW2Var = inputFile.getVar("VHM0_SW2");

		if(VHM0_SW2Var.isNull())
			return NC_ERR;
		VTM01_SW2Var = inputFile.getVar("VTM01_SW2");

		if(VTM01_SW2Var.isNull())
			return NC_ERR;
		VPEDVar = inputFile.getVar("VPED");

		if(VPEDVar.isNull())
			return NC_ERR;
		VTPKVar = inputFile.getVar("VTPK");

		if(VTPKVar.isNull())
			return NC_ERR;
		VMDR_SW2Var = inputFile.getVar("VMDR_SW2");

		if(VMDR_SW2Var.isNull())
			return NC_ERR;



		int currentNTIME;
		vector<size_t> startp,countp;
		for (currentNTIME=0; currentNTIME < NTIME; currentNTIME++)
		{

			startp.push_back(currentNTIME);
			startp.push_back(0);
			startp.push_back(0);

			countp.push_back(1);
			countp.push_back(NLAT);
			countp.push_back(NLON);

			sigHightVar.getVar(startp,countp,sig_hight_in);
			wavePeriodVar.getVar(startp,countp,wave_period_in);
			VTM02Var.getVar(startp,countp,VTM02_in);
			VMDRVar.getVar(startp,countp,VMDR_in);
			VSDXVar.getVar(startp,countp,VSDX_in);
			VSDYVar.getVar(startp,countp,VSDY_in);
			VHM0_WWVar.getVar(startp,countp,VHM0_WW_in);
			VTM01_WWVar.getVar(startp,countp,VTM01_WW_in);
			VMDR_WWVar.getVar(startp,countp,VMDR_WW_in);
			VHM0_SW1Var.getVar(startp,countp,VHM0_SW1_in);
			VTM01_SW1Var.getVar(startp,countp,VTM01_SW1_in);
			VMDR_SW1Var.getVar(startp,countp,VMDR_SW1_in);
			VHM0_SW2Var.getVar(startp,countp,VHM0_SW2_in);
			VTM01_SW2Var.getVar(startp,countp,VTM01_SW2_in);
			VMDR_SW2Var.getVar(startp,countp,VMDR_SW2_in);
			VPEDVar.getVar(startp,countp,VPED_in);
			VTPKVar.getVar(startp,countp,VTPK_in);
				

			for(int j = 0; j < NLON; j++) {
				for(int i = 0; i < NLAT; i++) { 
					if (sig_hight_in[0][i][j] < 0 || wave_period_in[0][i][j] < 0 ){
						wave_power[currentNTIME][i][j]=INVALID_VAL;
					}else{
						float sigHight=(float)sig_hight_in[0][i][j]*0.001;
						float wavePeriod=(float) wave_period_in[0][i][j]*0.001;
						float VTM02=(float)VTM02_in[0][i][j]*0.001;
						float VMDR=(float)VMDR_in[0][i][j]*0.01;
						float VSDX=(float)VSDX_in[0][i][j]*0.001;
						float VSDY=(float)VSDY_in[0][i][j]*0.001;
						float VHM0_WW=(float)VHM0_WW_in[0][i][j]*0.001;
						float VTM01_WW=(float)VTM01_WW_in[0][i][j]*0.001;
						float VMDR_WW=(float)VMDR_WW_in[0][i][j]*0.01;
						float VHM0_SW1=(float)VHM0_SW1_in[0][i][j]*0.001;
						float VTM01_SW1=(float)VTM01_SW1_in[0][i][j]*0.001;
						float VMDR_SW1=(float)VMDR_SW1_in[0][i][j]*0.01;
						float VHM0_SW2=(float)VHM0_SW2_in[0][i][j]*0.001;
						float VTM01_SW2=(float)VTM01_SW2_in[0][i][j]*0.001;
						float VMDR_SW2=(float)VMDR_SW2_in[0][i][j]*0.01;
						float VPED=(float)VPED_in[0][i][j]*0.01;
						float VTPK=(float)VTPK_in[0][i][j]*0.001;
						myfile << times[currentNTIME] << "," << currentNTIME << "," << argv[1] << "," << argv[2] << "," << sigHight << "," << wavePeriod << "," << VTM02 << "," << VMDR << "," <<  VSDX << "," <<  VSDY << "," <<  VHM0_WW << "," << VTM01_WW << "," <<  VMDR_WW << "," <<  VHM0_SW1 << "," <<  VTM01_SW1 << "," <<  VMDR_SW1 << "," <<  VHM0_SW2 << "," <<  VTM01_SW2 << "," <<  VMDR_SW2 << "," <<  VPED << "," <<  VTPK << "," << 0.49 * sigHight * sigHight * wavePeriod << std::endl;
					}
				}
			}

			while(!startp.empty()){
				startp.pop_back();
			}

			while(!countp.empty()){
				countp.pop_back();
			}
		}
		myfile.close();
	}catch(NcException& e) {
		e.what();
		cout<<"FAILURE"<<endl;
		return NC_ERR;
	}
	return 0;
} 