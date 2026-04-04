#ifndef SIMPARAMETERS_H
#define SIMPARAMETERS_H

struct SimParameters
{
    SimParameters()
    {
        timeStep = 0.001;
        NewtonMaxIters = 20;
        NewtonTolerance = 1e-8;
        
        gravityEnabled = true;
        gravityG = 1.0;                
    }

    double timeStep;
    double NewtonTolerance;
    int NewtonMaxIters;
    
    bool gravityEnabled;
    double gravityG;
};

#endif