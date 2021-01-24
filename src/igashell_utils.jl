export IGAShellConfigStateOutput

function generate_cohesive_oop_quadraturerule(zcoords::Vector{T}) where {T}
    
    ninterfaces = length(zcoords)-2
    ε = 1e-13

    points_top = Vector{Vec{1,T}}()
    weights_top = Vector{T}()

    points_bot = Vector{Vec{1,T}}()
    weights_bot = Vector{T}()
    #copy the oo-plane integration to each layer
    zcoords_interfaces = change_zcoord_range(zcoords)[2:end-1]
    for iinterface in 1:ninterfaces
        #Add the zcoord to the quadrature rule
        push!(points_top,  Vec( (zcoords_interfaces[iinterface] + ε, ) ))
        push!(weights_top, 1.0)

        #Add the zcoord to the quadrature rule
        push!(points_bot,  Vec( (zcoords_interfaces[iinterface] - ε, ) ))
        push!(weights_bot, 1.0)
        #push!(weights, qr.weights[qp])
    end
    return [QuadratureRule{1,RefCube,T}(weights_bot,points_bot), QuadratureRule{1,RefCube,T}(weights_top,points_top)]

end

#changes zcoord to -1 to 1
function change_zcoord_range(zcoords::Vector{T}) where T
    addon = (last(zcoords) + first(zcoords))/2
    scale = (last(zcoords) - first(zcoords))/2
    zcoords = (zcoords.-addon)/scale
    return zcoords
end


function initial_upgrade_of_dofhandler(dh::MixedDofHandler, igashell::IGAShell)

    instructions = Five.FieldDimUpgradeInstruction[]

    for (ic, cellid) in enumerate(igashell.cellset)

        cellnodes = igashell.cell_connectivity[:, ic]
        cellnode_states = @view adapdata(igashell).control_point_states[cellnodes]

        initial_cellnode_states = fill(LUMPED, length(cellnode_states))

        if cellnode_states != initial_cellnode_states
            ndofs = ndofs_per_cell(dh, cellid)

            instr = construct_upgrade_instruction(igashell, cellid, initial_cellnode_states, cellnode_states, zeros(Float64, ndofs), zeros(Float64, ndofs))
            push!(instructions, instr)
        end

    end
    return instructions
        
end

"""
IGAShellStressOutput
    Output of stresses at specific elements
"""
struct IGAShellStressOutput{P<:IGAShell,T} <: Five.AbstractOutput
    igashell::Base.RefValue{P}
    time_interval::T
    lastoutput::Base.RefValue{T}
    
    cellset::Vector{Int}

    statevar_history::Vector{Vector{ThroughThicknessStresses}}
    t::Vector{T}
end

function IGAShellStressOutput(part::Base.RefValue{I}; cellset::AbstractArray{Int}, interval::T) where {I<:IGAShell,T}
    @assert(!isempty(cellset))
    statevar_history = [ThroughThicknessStresses[] for _ in 1:length(cellset)]
    return IGAShellStressOutput{I,T}(part, interval, Base.RefValue(-1.0), collect(cellset), statevar_history, T[])
end

function update_output!(dh::JuAFEM.AbstractDofHandler, output::IGAShellStressOutput{I}, state::StateVariables{T}, ::SystemArrays) where {I,T}
    
    igashell = output.igashell[]
    dim_s = JuAFEM.getdim(igashell)

    nqp_inplane = getnquadpoints_inplane(igashell)
    nqp_outofplane = getnquadpoints_ooplane(igashell)
    nqp = getnquadpoints(igashell)

    oqr = get_oop_quadraturerule(intdata(igashell))

    qp_range = 1:nqp_inplane:nqp
    tts = ThroughThicknessStresses[]
    small_deformations = is_small_deformation_theory(layerdata(igashell))
    for (idx,cellid) in enumerate(output.cellset)
        local_id = findfirst((i)->i==cellid, igashell.cellset)

        ts = zeros(SymmetricTensor{2,3,T,6}, nqp_outofplane) #Through thickness stresses
        plot_coords = zeros(Vec{3,T}, nqp_outofplane)
        local_coords = zeros(Vec{3,T}, nqp_outofplane)
        zcoords = zeros(T, nqp_outofplane)
        #=for (i,matstate) in enumerate(state.partstates[cellid].materialstates[qp_range])
            ζ = oqr.points[i][1] 
            stress = getproperty(matstate, :σ)
            ts[i] = stress
            zcoords[i] = layerdata(igashell).total_thickness/2 * ζ #assume range from -1 to 1
            @show ζ
        end=#
        
        ic = cellid
        Ce = get_extraction_operator(intdata(igashell),cellid)

        _celldofs = zeros(Int, ndofs_per_cell(dh,cellid))
        celldofs!(_celldofs, dh, cellid)
        ue = state.d[_celldofs]
        
        #Coords of cell
        X = zeros(Vec{dim_s,T}, JuAFEM.nnodes_per_cell(igashell))
        JuAFEM.cellcoords!(X, dh, cellid)
        Xᵇ = IGA.compute_bezier_points(Ce, X)

        cv_sr = deepcopy(intdata(igashell).cell_values_sr) #cv_sr = intdata(igashell).cell_values_sr
        oop_values = _build_oop_basisvalue!(igashell, local_id, -1)
        IGA.set_bezier_operator!(cv_sr, Ce)

        set_oop_basefunctions!(cv_sr, oop_values)
        
        reinit!(cv_sr, Xᵇ)

        @assert(nqp_outofplane == getnquadpoints(cv_sr))
        iqp = 0
        for ilay in 1:nlayers(igashell)
            active_dofs = intdata(igashell).active_layer_dofs[ilay]
            ue_active = @view ue[active_dofs]
            for _ in 1:getnquadpoints_ooplane_per_layer(igashell)
                iqp +=1 
                σ, ξ, x_glob, x_loc= _eval_stress_center(cv_sr, igashell.layerdata.layer_materials[ilay], iqp, Xᵇ, ue_active, active_dofs, small_deformations)

                #zcoord = cv_sr.oqr.points[iqp][1] * layerdata(igashell).total_thickness/2

                local_coords[iqp] = x_loc
                ts[iqp] = σ
                plot_coords[iqp] = x_glob#layerdata(igashell).total_thickness/2 * ζ #assume range from -1 to 1
            end
        end
        push!(output.statevar_history[idx], ThroughThicknessStresses(ts, ts, local_coords, plot_coords))
    end

    push!(output.t, state.t)
end

function _eval_stress_center(cv::IGAShellValues{dim_s,dim_p,T}, material, qp, Xᵇ, ue, active_dofs, small_deformation_theroy::Bool) where {dim_s,dim_p,T}

    g = zeros(Vec{dim_s,T}, dim_s)

    x_glob = spatial_coordinate(cv, qp, Xᵇ)
    R = cv.R[qp]
    x_loc = R' ⋅ x_glob

    #@showm R
    ξ = get_qp_coord(cv, qp)

    #Covarient triad on deformed configuration
    for d in 1:dim_s
        g[d] = cv.G[qp][d] + function_parent_derivative(cv, qp, ue, d, active_dofs)
    end

    # Deformation gradient
    F = zero(Tensor{2,dim_s,T})
    for i in 1:dim_s
        F += g[i]⊗cv.Gᴵ[qp][i]
    end

    #Eval strains
    ɛ = symmetric(F) - one(SymmetricTensor{2,dim_s})

    #Eval strains
    if small_deformation_theroy
        ɛ = symmetric(F) - one(SymmetricTensor{2,dim_s})
        _̂ε = symmetric(R' ⋅ ɛ ⋅ R)
        _̂σ, ∂̂σ∂ɛ, new_matstate = constitutive_driver(material, _̂ε)
    else
        U = sqrt(tdot(F))
        E = symmetric(1/2 * (F' ⋅ F - one(F)))
        S, ∂S∂E, new_matstate = constitutive_driver(material, E)
        _̂σ = inv(det(F)) * U ⋅ S ⋅ U
    end
    
    x_loc_out = x_loc
    x_glob_out = x_glob
    σ_out = _̂σ
    if dim_s == 2
        x_loc_out = Vec{3}((x_glob[1], 0.0, x_glob[2]))
        x_glob_out = Vec{3}((x_loc[1], 0.0, x_loc[2]))
        σ_out = new_matstate.σ
    end

    return σ_out, ξ, x_glob_out, x_loc_out
end

"""
IGAShellRecovoredStressOutput
    Output
"""
struct IGAShellRecovoredStressOutput{dim_s, T, P<:IGAShell} <: Five.AbstractOutput
    igashell::Base.RefValue{P}
    time_interval::T
    lastoutput::Base.RefValue{T}
    
    cellset::Vector{Int}

    statevar_history::Vector{Vector{Vector{RecoveredStresses{T}}}} #vector of cell, timestep, zcoords
    cell_coordinate ::Vector{Vector{Vec{dim_s,T}}}
    t::Vector{T}
end

function IGAShellRecovoredStressOutput(part::Base.RefValue{I}; cellset::AbstractArray{Int}, interval::T) where {I<:IGAShell, T}
    @assert(!isempty(cellset))
    statevar_history = [RecoveredStresses{T}[] for _ in 1:length(cellset)]
    dim_s = JuAFEM.getdim(part[])
    cell_coordinate = [Vec{dim_s,T}[] for _ in 1:length(cellset)]
    return IGAShellRecovoredStressOutput{dim_s,T,I}(part, interval, Base.RefValue(-1.0), collect(cellset), statevar_history, cell_coordinate, T[])
end

function update_output!(dh::JuAFEM.AbstractDofHandler, output::IGAShellRecovoredStressOutput{dim_s,T}, state::StateVariables{T}, ::SystemArrays) where {dim_s,T}
    
    igashell = output.igashell[]
    cv_sr = intdata(igashell).cell_values_sr

    for (ic, cellid) in enumerate(output.cellset)

        #Get the mid coord of cell
        Ce = get_extraction_operator(intdata(igashell), cellid)

        X = zeros(Vec{dim_s,T}, JuAFEM.nnodes_per_cell(igashell))
        JuAFEM.cellcoords!(X, dh, cellid)
        Xᵇ = IGA.compute_bezier_points(Ce, X)
        
        reinit!(cv_sr, Xᵇ)
        
        #...midcoord:
        xᵐ = spatial_coordinate(cv_sr, 1, Xᵇ)

        local_cellid = cellid
        push!(output.statevar_history[ic], srdata(igashell).recovered_stresses[:, local_cellid])
        push!(output.cell_coordinate[ic], xᵐ)
    end
    push!(output.t, state.t)
end

"""
IGAShellIntegrationValuesOutput
    Output
"""
struct IGAShellIntegrationValuesOutput{P<:IGAShell,T} <: Five.AbstractOutput
    igashell::Base.RefValue{P}
    time_interval::T
    lastoutput::Base.RefValue{T}

    cellset::Vector{Int}

    values::Vector{Matrix{StressRecovoryIntegrationValues}} #one for each timestep;  qp x cellid
    t::Vector{T}
end

function IGAShellIntegrationValuesOutput(part::Base.RefValue{I}; cellset::AbstractArray{Int}, interval::T) where {I<:IGAShell, T}
    @assert(!isempty(cellset))
    statevar_history = Vector{Matrix{StressRecovoryIntegrationValues}}()
    return IGAShellIntegrationValuesOutput{I,T}(part, interval, Base.RefValue(-1.0), collect(cellset), statevar_history, T[])
end

function update_output!(dh::JuAFEM.AbstractDofHandler, output::IGAShellIntegrationValuesOutput, state::StateVariables{T}, ::SystemArrays) where T
    
    igashell = output.igashell[]
    dim_s = JuAFEM.getdim(igashell)

    npoints, nlay, ncells = size(srdata(igashell).integration_values)

    values = Matrix{StressRecovoryIntegrationValues}(undef, npoints*nlay, ncells )

    for (ic, cellid) in enumerate(output.cellset)

        local_cellid = findfirst((i)->i==cellid, igashell.cellset)

        values[:, ic] .= srdata(igashell).integration_values[:, :, local_cellid][:]

    end
    
    push!(output.values, values)
    push!(output.t, state.t)
end


#
generate_knot_vector(order::Int, ninterfaces::Int, nmultiplicity::Int) = generate_knot_vector(order, ninterfaces, fill(nmultiplicity, ninterfaces))

function generate_knot_vector(order::Int, ninterfaces::Int, nmultiplicity::Vector{Int})
    @assert(ninterfaces == length(nmultiplicity))

    kv = fill(-1.0, order+1)
    for i in 1:ninterfaces
        z = i/(ninterfaces+1)*2 - 1.0
        append!(kv, fill(z,nmultiplicity[i]))
    end
    append!(kv, fill(1.0,order+1))
    return kv
end

function generate_nmultiplicity_vector(state::CELLSTATE, ninterfaces::Int, order::Int) 
    if is_weak_discontiniuos(state)
        return digits(state.state2, base=2, pad=ninterfaces)*(order+1)
    elseif is_strong_discontiniuos(state)
        return digits(state.state2, base=2, pad=ninterfaces) .+ order
    elseif is_fully_discontiniuos(state)
        return fill(order+1, ninterfaces)
    elseif is_lumped(state)
        return fill(0,ninterfaces)
    elseif is_layered(state)
        return fill(order, ninterfaces)
    else
        error("bad state")
    end
end

function generate_out_of_plane_extraction_operators(_knot_vector::Vector{T}, order, new_knots::Vector{T}, multiplicity::Vector{Int}=fill(1, length(new_knots))) where {T}
    
    @assert(length(multiplicity) == length(new_knots))
    
    knot_vector = copy(_knot_vector)
    C = Diagonal(ones(length(knot_vector)-order-1))
    for (i, new_knot) in enumerate(new_knots)
        for _ in 1:multiplicity[i]
            C1, knot_vector = IGA.knotinsertion(knot_vector, order, new_knot)
            C = C1 * C
        end
    end
    return C
end

geometryobject(ip::Interpolation, indx::Int) = collect(1:getnbasefunctions(ip))
geometryobject(ip::Interpolation{2}, indx::FaceIndex) = 1:getnbasefunctions(ip)#JuAFEM.faces(ip)[indx[2]]
geometryobject(ip::Interpolation, indx::EdgeIndex) = JuAFEM.edges(ip)[indx[2]]
geometryobject(ip::Interpolation{1}, indx::VertexIndex) = JuAFEM.vertices(ip)[indx[2]]
geometryobject(ip::Interpolation{2}, indx::EdgeIndex) = JuAFEM.faces(ip)[indx[2]]
geometryobject(ip::Interpolation{2}, indx::EdgeInterfaceIndex) = JuAFEM.faces(ip)[indx[2]]
geometryobject(ip::Interpolation{2}, indx::VertexInterfaceIndex) = JuAFEM.vertices(ip)[indx[2]]

function active_basefunctions(field_dim::Int, indx::FaceIndex)
    indx[2] == 1 && return 1
    indx[2] == 2 && return field_dim
    error("Bad index")
end

function active_basefunctions(field_dim::Int, indx::Union{VertexInterfaceIndex, EdgeInterfaceIndex})
    indx[3] == 1 && return 1
    indx[3] == 2 && return field_dim
    error("Bad index")
end

function active_basefunctions(field_dim::Int, ::EdgeIndex)
    return 1:field_dim
end

function active_basefunctions(field_dim::Int, ::VertexIndex)
    return 1:field_dim
end

function active_basefunctions(field_dim::Int, ::Int)
    return 1:field_dim
end


#igashelldofs(dh::JuAFEM.AbstractDofHandler, igashell::IGAShell, index::GeometryObject) = celldofs(dh, index[1])[igashelldofs(igashell, index)]

function igashelldofs(igashell::IGAShell{dim_p,dim_s}, index::GeometryObject, components::Vector{Int}=collect(1:dim_s)) where {dim_p,dim_s}

    cellid = index[1]
    ip = getmidsurface_ip(layerdata(igashell))

    facepoints = geometryobject(ip, index)

    face_dofs = Int[]; currentdof = 1
    for (i, nodeid) in enumerate(cellconectivity(igashell, cellid))
        nodestate = get_controlpoint_state(adapdata(igashell), nodeid)
        nnodes_per_controlpoints = ndofs_per_controlpoint(igashell, nodestate) ÷ dim_s
        if i in facepoints
            for basefunc in active_basefunctions(nnodes_per_controlpoints, index)
                for d in components
                    push!(face_dofs, currentdof + (basefunc-1)*dim_s + d -1)
                end
            end
        end
        currentdof += nnodes_per_controlpoints*dim_s
    end

    return face_dofs
end


function extrapolate(y1,y2,x1,x2,x)
    return y1 + ((x-x1)/(x2-x1))*(y2-y1)
end#

"""
WIP, determine active dofs in each layer
"""

is_active_in_layer(ib::Int, ilay::Int, controlpoint_state::Val{LUMPED}, data::IGAShellData{dim_s,dim_p,T}) where {dim_s,dim_p,T} = true

function is_active_in_layer(ib::Int, ilay::Int, ::Val{LAYERED}, data::IGAShellData{dim_s,dim_p,T}) where {dim_s,dim_p,T}
    r = data.orders[dim_s]
    basefuncs_in_layer = (1:r+1) .+ (ilay-1)*(r+1) #not checked
    return ib in basefuncs_in_layer
end

function generate_active_dofs(data::IGAShellData{dim_s,dim_p,T}, controlpoint_states::Vector{CELLSTATE}) where {dim_s,dim_p,T}
    @assert length(controlpoint_states) == nnodes_per_cell(data)

    nlay = nlayers(data)
    active_layer_dofs = [Int[] for i in 1:nlay]
    
    dof_counter = 0
    for icp in 1:length(controlpoint_states) #loop over each inplane node
        cpstate = controlpoint_states[icp]
        for ilay in 1:nlay
            for ib in getnbasefunctions(cpstate) #loop over each ooplane node
                if is_active_in_layer(ib, cpstate, ilay, nlay)
                    for d in 1:dim_s
                        push!(active_layer_dofs[ilay], dof_counter + d)
                    end
                else
                    #Could proboboly break here...
                end
                dof_counter += dim_s
            end
        end
    end
end


"""
IGAShellBCOutput
    Output forces and deisplacement from global forcevectors and displacements vector
"""
struct IGAShellBCOutput{P<:IGAShell,T} <: Five.AbstractOutput
    igashell::Base.RefValue{P}
    components::Vector{Int}
end

function IGAShellBCOutput(; igashell::IGAShell, comps::Vector{Int})
    @assert( maximum(comps) <= 3 )
    T = Float64
    
    return IGAShellBCOutput{typeof(igashell),T}(Base.RefValue(igashell), comps)
end

function Five.build_outputdata(output::IGAShellBCOutput, set, dh::MixedDofHandler)
    return output
end

function Five.collect_output!(output::IGAShellBCOutput, state::StateVariables{T}, faceset, globaldata) where T
    
    dh = globaldata.dh 
    igashell = output.igashell[]
    dim_s = JuAFEM.getdim(igashell)

    nnodes = JuAFEM.nnodes_per_cell(igashell)
    X = zeros(Vec{dim_s,T}, nnodes)

    alldofs = Int[]
    maxu = 0.0
    for faceidx in faceset
        @show faceidx
        cellid = faceidx[1]
        local_cellid = findfirst((i)->i==cellid, igashell.cellset)
        
        _celldofs = zeros(Int, JuAFEM.ndofs_per_cell(dh, cellid))
        JuAFEM.celldofs!(_celldofs, dh, cellid)
        
        cellstate = getcellstate(adapdata(igashell), local_cellid)
        #
        # This works if the face/edge is on the boundary of the volume
        # However, if it is an internal face/edge, it will not work
            #=
            local_dofs = igashelldofs(igashell, faceidx, output.components)
            facedofs = _celldofs[local_dofs]
            append!(alldofs, facedofs)=#

        #
        # This approach works generally. Se what basefunctions are non-zero, and take the corresponding dofs
        Ce = get_extraction_operator(intdata(igashell), local_cellid)
        JuAFEM.cellcoords!(X, dh, cellid)
        Xᵇ = IGA.compute_bezier_points(Ce, X)

        cv = build_facevalue!(igashell, faceidx)
        IGA.set_bezier_operator!(cv, Ce)

        active_layerdofs = build_active_layer_dofs(igashell, cellstate)
        reinit_midsurface!(cv, Xᵇ)

        for ilay in 1:nlayers(igashell)

            reinit_layer!(cv, ilay)

            layerdofs = _celldofs[active_layerdofs[ilay]]
            ue = state.d[layerdofs]

            for qp in 1:getnquadpoints_per_layer(cv)
               
                u = function_value(cv, qp, ue)
                for d in output.components
                    maxu = max(maxu, abs(u[d]))
                end
                
                #Loop through all basefunctions
                for dof in 1:getngeombasefunctions_per_layer(cv)
                    #Check if it is not zero
                    if !(basis_value(cv, qp, (dof-1)*dim_s + 1)[1] ≈ 0.0)
                        for d in output.components
                            push!(alldofs, layerdofs[(dof-1)*dim_s + d])
                        end
                    end
                end
            end

        end

    end

    unique!(alldofs)

    return (forces        = sum(state.system_arrays.fⁱ[alldofs]),
            displacements = maxu) # maximum(abs.(state.d[alldofs]))

end

"""
IGAShellConfigStateOutput
"""

struct IGAShellConfigStateOutput <: Five.AbstractOutput end

"""
IGAShellTractionForces
    Output traction forces from specific cells
"""
const InterfaceTractionJumpPosition{dim_s,T} = Tuple{Vec{dim_s,T}, Vec{dim_s,T}, Vec{dim_s,T}}

struct IGAShellTractionForces{P<:IGAShell,dim_s,T} <: Five.AbstractOutput
    igashell::Base.RefValue{P}
    time_interval::T
    lastoutput::Base.RefValue{T}

    interfaceidxs::Vector{Tuple{Int,Int}} #cellid, interfaceidx

    interfaceforce::Vector{Vector{Vector{InterfaceTractionJumpPosition{dim_s,T}}}} #time, cell, qp
    t::Vector{T}
end

function IGAShellTractionForces(part::Base.RefValue{I}; interfaceidxs::Vector{Tuple{Int,Int}}, interval::T) where {I<:IGAShell, T}
    dim_s = JuAFEM.getdim(part[])

    interfaceforce = Vector{Vector{InterfaceTractionJumpPosition{dim_s,T}}}()
    return IGAShellTractionForces{I,dim_s,T}(part, interval, Base.RefValue(-1.0), interfaceidxs, interfaceforce, T[])
end

function update_output!(dh::JuAFEM.AbstractDofHandler, output::IGAShellTractionForces, state::StateVariables{T}, system_arrays::SystemArrays) where T
    
    igashell = output.igashell[]
    dim_s = JuAFEM.getdim(igashell)
    
    celltractions = Vector{Vector{InterfaceTractionJumpPosition{dim_s,T}}}()
    for (cellid, interfaceidx) in output.interfaceidxs

        #Get cellvalues
        cv, _ = build_cohesive_cellvalue!(igashell, cellid) 
            
        #Bezier operator
        Ce = get_extraction_operator(intdata(igashell),cellid)
        IGA.set_bezier_operator!(cv, Ce)
    
        #Coords of cell
        X = zeros(Vec{dim_s,T}, JuAFEM.nnodes_per_cell(igashell))
        JuAFEM.cellcoords!(X, dh, cellid)
        Xᵇ = IGA.compute_bezier_points(Ce, X)

        #Spatial coord and traction
        tractions = InterfaceTractionJumpPosition{dim_s,T}[]
        for iqp in 1:getnquadpoints_per_interface(igashell)
            t = state.partstates[cellid].interfacestates[iqp,interfaceidx].t
            J = state.partstates[cellid].interfacestates[iqp,interfaceidx].J
            x = spatial_coordinate(cv, iqp, Xᵇ)
            push!(tractions, (t,J,x))
        end
        push!(celltractions, tractions)
    end

    push!(output.interfaceforce, celltractions)
    push!(output.t, state.t)
end

"""
Returns the vertex id below,
4__________3
|          |
|__________|
1          2
, given the vertex id 1,3,7 or 9 from the bezierelement
7____8_____9
|          |
4    5     6
|__________|
1    2     3
"""
function vertex_converter(igashell::IGAShell{2,3}, v::Int)
    if v == 1
        return 1
    elseif v == igashell.layerdata.orders[1]+1
        return 2
    elseif v == JuAFEM.nnodes_per_cell(igashell)
        return 3
    elseif v == JuAFEM.nnodes_per_cell(igashell)-igashell.layerdata.orders[1]
        return 4
    else
        error("Bad vertex")
    end
end

function vertex_converter(igashell::IGAShell{1,2}, v::Int)
    if v == 1
        return 1
    elseif v == JuAFEM.nnodes_per_cell(igashell)
        return 2
    else
        error("Bad vertex")
    end
end


"""
_convert_2_3dstate
    Takes a tensor i 2d and convertes into a stress state in 3d
    Assume plane stress
"""
function _convert_2_3dstate(σ::SymmetricTensor{2,2,T}) where T
    return SymmetricTensor{2,3,T,6}((σ[1,1], T(0.0), σ[1,2], T(0.0), T(0.0), σ[2,2]))
end