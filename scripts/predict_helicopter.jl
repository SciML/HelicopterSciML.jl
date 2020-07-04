using Pkg
cd(@__DIR__)
Pkg.activate("."); Pkg.instantiate()

using Plots; gr()
using LaTeXStrings
using DataFrames
using CSV
using OrdinaryDiffEq
using LinearAlgebra
using DataInterpolations
using BenchmarkTools
using DiffEqFlux
using Optim
using Flux
using Zygote
using ApproxFun

# Model parameters
rc = 1.5e-2 # m
h = 0.5e-2 # m
m = 0.5 # kg
Jbc = 15e-3 # kg.m2
Js = 5e-3 # kg.m2
g = 9.81 # m/s2
K_th_th = 55e-3 # Nm/V
K_th_psi = 5e-3 # Nm/V
K_psi_th = 0.15 # Nm/V
K_psi_psi = 0.2 # Nm/V
D_th = 1e-2 # Nm/(rad/s)
D_psi = 8e-2 # Nm/(rad/s)

# Graphics
LW1 = 2.5
LW2 = 1.5
LS1 = :solid
LS2 = :dot
LS3 = :dash
LC1 = :blue
LC2 = :red
LC3 = :green
LC4 = :black
LA1 = 1.0
LA2 = 0.7
LA3 = 0.3

# Parse data from table
df=CSV.read("/Users/diogo/UROP/Hybrid_Helicopter_model/data/Lab-Helicopter_Experimental-data.csv")
ENV["COLUMNS"]=100 # change the number of IJulia columns from default 80 to 100
df[1:5,:]
u_th = df[:,1]
u_psi = df[:,2]
theta = df[:,3]
psi = df[:,4]
dt = df[1,5]
tm = 0:dt:(length(psi)-1)*dt
tspan = (0.0, (length(psi)-1)*dt)
Nd = length(tm);
timeseriesdata = [theta';psi']

A = fill(FourierBasis(2), 6)
nn1 = TensorLayer(A, 2)
N1 = size(nn1.p)[1]
α = [-0.05712984834024423, -0.043425838041362616, -0.039491674348934955, 0.021965193327086824, 0.02325504921460965, 0.06213632889050528, -0.024592635430154844, -0.023456367225882492, -0.038315313635125234, -0.03724487007644209, 0.056930204755958815, 0.06009194061881117, 0.023921528544068718, 0.031184849334947806, -0.016706564345846196, 0.058481480251928876, -0.019767993044654295, 0.007465374144667601, 0.0001978884910513552, -0.012003256233740982, -0.010185415176279092, -0.061904979432675714, 0.01726548206019831, 0.002797592993383623, 0.015681312250059656, 0.04229278027253888, -0.04648580659148829, -0.058887747690087824, -0.06678831391728852, -0.0485370223800822, 0.000997581327942135, -0.017321695255257744, 0.0044523000280360285, -0.015705179251052612, 0.030105997917661248, 0.05396291339415509, 0.045888363929615636, 0.04717859426232344, 0.03840560012117353, 0.017766851339116835, 0.01019997462362888, 0.06151200389544084, -0.046775339185423336, -0.0421889300228381, -0.04435200979608834, -0.04301086175837168, 0.004469198394506686, -0.05066990175478263, 0.008760108397509979, -0.027028387780466054, -0.008360104289194696, 0.013711919078030921, 0.008661479530592911, 0.04231383800963808, -0.0050224363479699895, 0.013443794725208827, -0.008435978451943363, -0.015972469491542362, 0.014921741156510081, 0.02956894022038565, 0.055598236592215, 0.03726922970875653, 0.012159970589412449, 0.01713207816959305, -0.013841217575831689, 0.02818252004965422, -0.04651282869545733, -0.04713460729855985, -0.03982621162330606, -0.05467619435849095, 0.0041457665581151135, 0.0152698280670041, 0.01366093953190812, 0.02282177655258769, -0.024176202666935074, -0.05600131656498439, -0.00865137968516389, -0.042159873813811984, -0.0119152540783539, 0.025528833750960975, -0.024269599721797437, -0.020819463844186654, 0.042519963768813826, 0.02296053173375865, 0.011813014703805821, -0.030080393071379663, -0.014297959256416023, 0.004762416266580192, -0.04991265218830253, -0.026293762332667715, 0.030681154263069974, 0.05716061329827961, 0.0373965534306748, 0.048641127902609486, 0.023723289600094743, 0.022140039301380633, 0.02363002332016135, 0.036221657777115286, -0.03258297605532347, -0.03259789261146117, -0.02402746425611025, -0.013990198532076448, -0.016180517182257607, -0.021805632162810992, -0.005266698787528003, -0.015436439191717558, 0.0466036546003419, 0.05165981970758245, 0.010921347698531397, 0.05383803771783719, 0.0166834008165448, -0.014121442266036696, 0.039133222513240876, -0.0038926495906029257, -0.0317213415580547, -0.009682472467768094, -0.030954202645652282, 0.007394328518238946, 0.017395501252344358, -0.010101551899406288, 0.045693175555893535, 0.017312150380196505, -0.052092267044998336, -0.045678875675782764, -0.0354927840722541, -0.06403482098166911, -0.03591588723194942, -0.024257818595279815, -0.0484927670160119, -0.014423345852937482, 0.04699427853513137, 0.032870299358240616, 0.03411438561169296, 0.005847570916566264]

function helicopter_uode(dx,x,Ps,P0,u,p,t)
# P0 = (rc=rc,h=h,m=m,Jbc=Jbc,Js=Js,g=g,K_th_th=K_th_th,K_th_psi=K_th_psi,K_psi_th=K_psi_th,K_psi_psi=K_psi_psi,D_th=D_th,D_psi=D_psi);
    th = x[1]
    psi = x[2]
    w_th = x[3]
    w_psi = x[4]

    U = [u.th(t),u.psi(t)]
    #
    M11 = Ps[4]*P0.Jbc + Ps[3]*P0.m*((Ps[1]*P0.rc)^2+(Ps[2]*P0.h)^2)
    M22 = Ps[4]*P0.Jbc*cos(th)^2 + Ps[3]*P0.m*(Ps[1]*P0.rc*cos(th)-Ps[2]*P0.h*sin(th))^2 + Ps[5]*P0.Js
    #
    dLdth = -w_psi^2*(((Ps[4]*P0.Jbc)+(Ps[3]*P0.m)*((Ps[1]*P0.rc)^2-(Ps[2]*P0.h)^2))*cos(th)*sin(th)-(Ps[1]*P0.rc)*(Ps[2]*P0.h)*(cos(th)^2-sin(th)^2)) - (Ps[3]*P0.m)*(Ps[6]*P0.g)*((Ps[1]*P0.rc)*cos(th) - (Ps[2]*P0.h)*sin(th))
    #
    dp_psi_dth = -2*(((Ps[4]*P0.Jbc)+(Ps[3]*P0.m)*((Ps[1]*P0.rc)^2-(Ps[2]*P0.h)^2))*cos(th)*sin(th) + (Ps[3]*P0.m)*(Ps[1]*P0.rc)*(Ps[2]*P0.h)*(cos(th)^2-sin(th)^2))*w_psi
    #
    F_th = Ps[7]*P0.K_th_th*U[1] - Ps[8]*P0.K_th_psi*U[2] - P0.D_th*(w_th) + nn1([p[1]*U[1], p[2]*U[2], p[3]*th, p[4]*psi, p[5]*w_th, p[6]*w_psi], p[7:end])[1]
    F_psi = Ps[9]*P0.K_psi_th*U[1] - Ps[10]*P0.K_psi_psi*U[2] - P0.D_psi*(w_psi) + nn1([p[1]*U[1], p[2]*U[2], p[3]*th, p[4]*psi, p[5]*w_th, p[6]*w_psi], p[7:end])[2]

    #
    dx[1] = w_th
    dx[2] = w_psi
    dx[3] = (dLdth + F_th)/M11
    dx[4] = (-dp_psi_dth*w_th + F_psi)/M22
end

P0 = (rc=rc,h=h,m=m,Jbc=Jbc,Js=Js,g=g,K_th_th=K_th_th,K_th_psi=K_th_psi,K_psi_th=K_psi_th,K_psi_psi=K_psi_psi,D_th=D_th,D_psi=D_psi)
Ps = fill(1.0,length(P0))
#
th0 = theta[1]
psi0 = psi[1]
w_th0 = (theta[2]-theta[1])/dt
w_psi0 = (psi[2]-psi[1])/dt
#
x0 = [th0,psi0,w_th0,w_psi0]
U_th = ZeroSpline(u_th,tm)
U_psi = ZeroSpline(u_psi,tm)
u_t = (th=U_th, psi=U_psi)
helicopter! = (dx,x,p,t) -> helicopter_uode(dx,x,Ps,P0,u_t,p,t)

prob = ODEProblem(helicopter!,x0,tspan,p=nothing)
function uode_cost(p)
    #
    sol = solve(prob, p=p, AutoTsit5(TRBDF2(autodiff=false)), saveat=dt)
    #
    return norm(sol[1:2,1:end]-timeseriesdata[:,1:size(sol,2)])^2/Nd
    #
end

rad2deg(t,u) = (t,u*180/pi)

callback = function (p, l)
    @show p, l
    prob = ODEProblem(helicopter!,x0,tspan,p)
    sol = solve(prob, AutoTsit5(TRBDF2(autodiff=false)), saveat=dt)
    pl = plot(sol,vars=(rad2deg,0,2),lw=LW1,lc=LC1)
    plot!(pl,tm,psi*180/pi,lw=LW1,ls=LS2,lc=LC2)
    plot!(pl,xlim = tspan)
    plot!(pl)
    display(pl)
    return false
end
res = DiffEqFlux.sciml_train(uode_cost, α, ADAM(0.01), cb = callback, maxiters = 150)

p = res.minimizer
