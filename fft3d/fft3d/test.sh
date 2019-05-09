#!/usr/bin/bash
#env CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 bin/host -n 16 -i 100 -o output.csv
#env CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 bin/host -n 32 -i 100 -o output.csv
#env CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 bin/host -n 64 -i 100 -o output.csv

bin/host -n 16 -i 100 -o output.csv
bin/host -n 32 -i 100 -o output.csv
bin/host -n 64 -i 100 -o output.csv

bin/host -n 16 -i 100 -b -o output_inv.csv
bin/host -n 32 -i 100 -b -o output_inv.csv
bin/host -n 64 -i 100 -b -o output_inv.csv
