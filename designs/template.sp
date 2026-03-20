* Auto-generated

.param WN1={WN1}
.param WP1={WP1}
.param WN2={WN2}
.param WP2={WP2}
.param L1={L1}
.param L2={L2}

.param VDD={VDD}
.param TEMP={TEMP}

.include "{model}"
.include "{pvt_corner}"
.include "{skew_corner}"


* All measurements are defined in 2inv.sp
.include "2inv.sp"