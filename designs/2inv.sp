* ============================================
* Cascaded Inverter Testbench for ML Dataset
* All node currents and voltages measurable
* ============================================

.option post
.option nomod
.option accurate
.option method=gear

* --------- Subcircuits for current measurement ---------

.subckt PMOS_MEAS D G S B Dmeas 0 PARAMS: WP=1u L=100n
Vmeas_p D Dmeas 0
M1 Dmeas G S B pmos W={WP} L={L}
.ends PMOS_MEAS

.subckt NMOS_MEAS D G S B Dmeas 0 PARAMS: WN=1u L=100n
Vmeas_n D Dmeas 0
M1 Dmeas G S B nmos W={WN} L={L}
.ends NMOS_MEAS

.subckt C_MEAS N1 N2 N1meas 0 PARAMS: CVAL=1f
Vmeas_c N1 N1meas 0
C1 N1meas N2 {CVAL}
.ends C_MEAS

* --------- Driver Inverter ---------
XMP1 target_int in vdd vdd target_mp1 0 PMOS_MEAS PARAMS: WP={WP1} L={L1}
XMN1 target_int in 0 0 target_mn1 0 NMOS_MEAS PARAMS: WN={WN1} L={L1}
Vmeas_target target target_int 0V

* --------- Load Inverter ---------
XMP2 out_meas target vdd vdd out_mp2 0 PMOS_MEAS PARAMS: WP={WP2} L={L2}
XMN2 out_meas target 0 0 out_mn2 0 NMOS_MEAS PARAMS: WN={WN2} L={L2}
Vmeas_out out out_meas 0

* --------- Load Cap (optional realism) ---------
XCload out 0 out_cload 0 C_MEAS PARAMS: CVAL=5f

* --------- Analysis ---------
.tran 1p 5n

* --------- Model Files ---------
.include models/22nm_HP.pm

* --------- Output ---------
.control
run
print v(vdd) v(in) v(out) v(target)
print vmeas_vdd#branch vmeas_in#branch vmeas_out#branch vmeas_target#branch
print tran I(vmeas_vdd) I(vmeas_in) I(vmeas_out) I(vmeas_target)
.endc

* --------- Supplies ---------
VDD vdd_src 0 {VDD}
Rsupply vdd_meas vdd_src 0.01
Vmeas_vdd vdd vdd_meas 0

* --------- Input ---------
Vin in_src 0 PULSE(0 {VDD} 0 20p 20p 1n 2n)
Rin in_meas in_src 0.01
Vmeas_in in in_meas 0

* --------- Ground connection (forced to 0)
* gnd is now directly connected to SPICE ground
* Remove Rgnd, connect gnd to 0 with a wire
* In SPICE, just use the same node name for both:
* All references to 'gnd' are now node 0
