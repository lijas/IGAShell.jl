using Five
using IgAShell

#Dimension
DIM = 3
NELX = 31
NELY = 15
ORDERS = (3,3,2)
@assert(isodd(NELY) && isodd(NELX)) #So outputed cell for stresses is in the center

#Geometry
P = -0.1
R1 = 25.0
R2 = 10.0
h = 0.8
a = R1*pi/2 #arclength

#
angles = deg2rad.([0.0, 90.0, 45.0, -45.0])

data = ProblemData(
    dim = DIM,
    tend = 1.0,
    adaptive = true
)

interfacematerial = MatCZBilinear(
    K    = 1.0e5,
    Gᴵ   = (0.5, 0.5, 0.5),
    τᴹᵃˣ = (50.0, 50.0, 50.0),
    η    = 1.6
) 

material(_α) = 
MatTransvLinearElastic(
    E1 = 100e3,
    E2 = 10e3,
    ν_12 = 0.25,
    G_12 = 5e3,
    α = _α
) 

layermats = [material(α) for α in angles]

#
nurbsmesh = IgAShell.IGA.generate_doubly_curved_nurbsmesh((NELX,NELY), ORDERS[1:2], r1 = R1, r2 = R2, α1 = pi/2, α2 = deg2rad(95))
data.grid = IgAShell.IGA.convert_to_grid_representation(nurbsmesh)

#Setsk and variables
nlayers = length(angles)
ninterfaces = nlayers - 1
addedgeset!(data.grid, "clamped", (x)-> x[3]<0.01)
addedgeset!(data.grid, "force",   (x)-> x[2]>-0.01)
postcell = ceil(Int, 0.5*(NELX*NELY))
partset1 = collect(1:NELX*NELY)

#
cellstates = [IgAShell.LUMPED for i in 1:getncells(data.grid)]
interface_damage = zeros(ninterfaces, getncells(data.grid))

#IGAshell data
igashelldata = 
IgAShell.IGAShellData(;
    layer_materials           = layermats,
    interface_material        = interfacematerial,
    viscocity_parameter       = 0.0,
    orders                    = ORDERS,
    knot_vectors              = nurbsmesh.knot_vectors,
    thickness                 = h,
    initial_cellstates        = cellstates,
    initial_interface_damages = interface_damage,
    nqp_inplane_order         = 4,
    nqp_ooplane_per_layer     = 2,
    adaptable                 = false,
    small_deformations_theory = true,
    LIMIT_UPGRADE_INTERFACE   = 0.04,
    nqp_interface_order       = 4
)  

igashell = 
IgAShell.IGAShell(
    cellset = partset1, 
    connectivity = reverse(nurbsmesh.IEN, dims=1), 
    data = igashelldata
) 
push!(data.parts, igashell)

#
data.output[] = Output(
    interval = 0.0,
    runname = string("sr_doubly_curved_", NELX,"x",NELY,"_ooporder",ORDERS[3]),
    savepath = "./"
)

#
etf = IGAShellWeakBC( 
    set = getedgeset(data.grid, "clamped"),
    func = (x,t) -> zeros(DIM), 
    comps = 1:DIM,
    igashell = igashell, 
    penalty = 1e9
)
push!(data.constraints, etf)

#Force
forcevec(x,t) = Vec{3}((0.0, 0.0, t*P/(a*h)))
#forcevec(x,t) = Vec{3}((0.0, 0.0, x[1] * t*P/(a*h)))
etf = IGAShellExternalForce(
    set = getedgeset(data.grid, "force"), 
    func = (x,t) -> forcevec(x,t),
    igashell = igashell
)
push!(data.external_forces, etf)

#
vtkoutput = VTKCellOutput(
    type = IGAShellConfigStateOutput()
)
Five.push_vtkoutput!(data.output[], vtkoutput)

output = OutputData(
    type = IGAShellStressOutput(
        igashell = igashell,
    ),
    interval = 0.0,
    set      = [postcell]
)
data.outputdata["Stress at 50%"] = output

#
output = OutputData(
    type = IGAShellRecovoredStressOutput(
        igashell = igashell,
    ),
    interval = 0.0,
    set      = [postcell]
)
data.outputdata["RS at 50%"] = output

#
state, globaldata = build_problem(data) do dh, parts, dbc
    instructions = IgAShell.initial_upgrade_of_dofhandler(dh, igashell)
    Five.update_dofhandler!(dh, StateVariables(Float64, ndofs(dh)), instructions)
    
    alldofs = collect(1:ndofs(dh))
    JuAFEM.copy!!(dbc.free_dofs, alldofs)
end

solver = NewtonSolver(
    Δt0 = 1.0, 
    Δt_max = 1.0, 
    tol=1e-7
)


output = solvethis(solver, state, globaldata)

if true
    using Plots; pyplot(); PyPlot.pygui(true)
    using JLD2; using FileIO

    σ₀ = abs(P/(a*h))
    
    σᶻˣ_vec = getproperty.(output.outputdata["RS at 50%"].data[end][1], :σᶻˣ) ./ σ₀
    σᶻʸ_vec = getproperty.(output.outputdata["RS at 50%"].data[end][1], :σᶻʸ) ./ σ₀
    σᶻᶻ_vec = getproperty.(output.outputdata["RS at 50%"].data[end][1], :σᶻᶻ) ./ σ₀
    ζ_vec = getproperty.(output.outputdata["RS at 50%"].data[end][1], :ζ)
    
    fig = plot(reuse=false, layout = (2,3))
    plot!(fig[4], σᶻᶻ_vec, ζ_vec, label = "Lumped")
    plot!(fig[5], σᶻʸ_vec, ζ_vec, label = "Lumped")
    plot!(fig[6], σᶻˣ_vec, ζ_vec, label = "Lumped")
    
    #
    σxx_lumped = getindex.(output.outputdata["Stress at 50%"].data[end][1].stresses, 1, 1)
    σyy_lumped = getindex.(output.outputdata["Stress at 50%"].data[end][1].stresses, 2, 2)
    σxy_lumped = getindex.(output.outputdata["Stress at 50%"].data[end][1].stresses, 1, 2)
    zcoords = getindex.(output.outputdata["Stress at 50%"].data[end][1].local_coords, 3)

    plot!(fig[1], σxx_lumped, zcoords, label = "Lumped")
    plot!(fig[2], σyy_lumped, zcoords, label = "Lumped")
    plot!(fig[3], σxy_lumped, zcoords, label = "Lumped")
    
    #Reference
    #σ_dr_xx = getindex.(data_dr.stresses, 1, 3)./ σ₀
    #σ_dr_yy = getindex.(data_dr.stresses, 1, 3)./ σ₀
    #σ_dr_xy = getindex.(data_dr.stresses, 1, 3)./ σ₀

    #REFERENCE     
    #SCALED WITH ./ σ₀
    #TWIST
    σ_dr_zx = [-0.017456545285163178, -0.08009531103996745, -0.15656468560999176, -0.211161449879248, -0.22361224710426564, -0.21817530746427527, -0.21180097289075042, -0.20745085551624756, -0.2045029048065384, -0.20745655218212597, -0.2386773032655793, -0.28339033775746003, -0.2819116789974262, -0.21514154061030744, -0.11362290026061687, -0.02491613000251832]
    σ_dr_zy = [-0.01875293096798717, -0.03372860837518501, -0.04406828166079741, -0.04483687130904259, -0.11872422742297192, -0.3030089110961427, -0.3316979472678521, -0.19285787493738588, -0.12196089682433287, -0.08354764100352022, -0.048100070279557504, -0.032261893277322824, -0.025869361860650482, -0.014940360234569778, -0.0021073814614485176, 0.006574950624109785]
    σ_dr_zz = [-0.007654655792413438, -0.08593045229416249, -0.1838638203087384, -0.2556437990235454, -0.29856228011087454, -0.39165533287970816, -0.5240335607827316, -0.63286121469756, -0.6215089368270628, -0.5293585001056942, -0.563979289422609, -0.7086456919681258, -0.7566138882490854, -0.6575165298922369, -0.39348689881430704, -0.08769617981353893]
    z_dr = [-0.38611363115940733, -0.3339981043584821, -0.2660018956415158, -0.2138863688405941, -0.18611363115940094, -0.13399810435848636, -0.06600189564151293, -0.013886368840594798, 0.013886368840594798, 0.06600189564152004, 0.1339981043584899, 0.1861136311594045, 0.21388636884059764, 0.2660018956415193, 0.33399810435848565, 0.38611363115940733]

    #ZFORCE
    σ_dr_zy = [-0.0010673747226584232, -0.0006599347127446362, 0.0004606505957739027, 0.0017893863706277206, 0.0032975014213682177, 0.009583389537997375, 0.02291974668148687, 0.03706654579214346, 0.042534047312958394, 0.059646380785515624, 0.10938690786279201, 0.16834723741872323, 0.18066172059305927, 0.14756176906517401, 0.08398842292486668, 0.019758845634705945]
    σ_dr_zx = [-0.022986783384666268, -0.0973272596374496, -0.1941625513075552, -0.26828181215163216, -0.2926452531167654, -0.30799882743434026, -0.32322311616296917, -0.33132128007440415, -0.33405477660850064, -0.32510319056937115, -0.2834554606570657, -0.22865589148371154, -0.20401233707896985, -0.16757085402216457, -0.09651104588164434, -0.024085543159153885]
    σ_dr_zz = [0.013452932630954841, 0.12990982206991136, 0.2839695837052828, 0.40317811154185407, 0.568064483452593, 0.9435596416781139, 1.1500917698166384, 1.0887640048962224, 1.0444733061460556, 1.0085340941692489, 0.8805581435575113, 0.7191659429327077, 0.6437786263049335, 0.5120162836455201, 0.2822739618044135, 0.06065054258671203]
    z_dr = [-0.38611363115940733, -0.3339981043584821, -0.2660018956415158, -0.2138863688405941, -0.18611363115940094, -0.13399810435848636, -0.06600189564151293, -0.013886368840594798, 0.013886368840594798, 0.06600189564152004, 0.1339981043584899, 0.1861136311594045, 0.21388636884059764, 0.2660018956415193, 0.33399810435848565, 0.38611363115940733]

    #plot!(fig[1], σ_dr_xx./ σ₀, z_dr, label = "Ref")
    #plot!(fig[2], σ_dr_yy, z_dr, label = "Ref")
    #plot!(fig[3], σ_dr_xy, z_dr, label = "Ref")
    plot!(fig[4], σ_dr_zz, z_dr, label = "Ref")
    plot!(fig[5], σ_dr_zy, z_dr, label = "Ref")
    plot!(fig[6], σ_dr_zx, z_dr, label = "Ref")

end



