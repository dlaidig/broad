//=====================================================================================================
// MahonyAHRS.h
//=====================================================================================================
//
// Madgwick's implementation of Mayhony's AHRS algorithm.
// See: http://www.x-io.co.uk/node/8#open_source_ahrs_and_imu_algorithms
//
// Date			Author			Notes
// 29/09/2011	SOH Madgwick    Initial release
// 02/10/2011	SOH Madgwick	Optimised for reduced CPU load
//
//=====================================================================================================
#ifndef MahonyAHRS_h
#define MahonyAHRS_h

//----------------------------------------------------------------------------------------------------
// Variable declaration

extern volatile float twoKp;			// 2 * proportional gain (Kp)
extern volatile float twoKi;			// 2 * integral gain (Ki)
extern volatile float q0, q1, q2, q3;	// quaternion of sensor frame relative to auxiliary frame

//---------------------------------------------------------------------------------------------------
// Function declarations

class MahonyAHRS {
public:
    MahonyAHRS(float Kp, float Ki, float sampleFreq);
    void update(float gx, float gy, float gz, float ax, float ay, float az, float mx, float my, float mz);
    void updateIMU(float gx, float gy, float gz, float ax, float ay, float az);

    float twoKp;									// 2 * proportional gain (Kp)
    float twoKi;									// 2 * integral gain (Ki)
    float q0, q1, q2, q3;					        // quaternion of sensor frame relative to auxiliary frame
    float integralFBx,  integralFBy, integralFBz;	// integral error terms scaled by Ki
    float sampleFreq;
};

#endif
//=====================================================================================================
// End of file
//=====================================================================================================
