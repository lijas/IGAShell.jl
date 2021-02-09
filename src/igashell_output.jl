export IGAShellConfigStateOutput, IGAShellStressOutput, IGAShellBCOutput, IGAShellRecovoredStressOutput

"""
IGAShellStressOutput
    Output of stresses at specific elements
"""
struct IGAShellStressOutput{P<:IGAShell} <: Five.AbstractOutput
    igashell::Base.RefValue{P}
end

function IGAShellStressOutput(; igashell::IGAShell)
    return IGAShellStressOutput(Base.RefValue(igashell))
end

function Five.build_outputdata(output::IGAShellStressOutput, set, dh::MixedDofHandler)
    @assert(eltype(set) == Int) # Only accept cellids
    return output
end

function Five.collect_output!(output::IGAShellStressOutput, state::StateVariables{T}, cellset, globaldata) where T
    
    #Extract some variables
    igashell = output.igashell[]
    dh = globaldata.dh
    dim_s = JuAFEM.getdim(igashell)
    small_deformations = is_small_deformation_theory(layerdata(igashell))

    #pre-allocate variables
    nnodes = JuAFEM.nnodes_per_cell(igashell)
    X = zeros(Vec{dim_s,T}, nnodes)

    #Variables related to number of gausspoints
    nqp_inplane     = getnquadpoints_inplane(igashell)
    nqp_outofplane = getnquadpoints_ooplane(igashell)
    nqp            = getnquadpoints(igashell)
    oqr            = get_oop_quadraturerule(intdata(igashell))

    #Collect stresses throught thickness
    ttstresses = ThroughThicknessStresses[]
    for (idx,cellid) in enumerate(cellset)
        
        ts = ThroughThicknessStresses()

        #Get cellstate
        local_id = findfirst((i)->i==cellid, igashell.cellset)
        cellstate = getcellstate(igashell, local_id)

        Ce = get_extraction_operator(intdata(igashell), local_id)

        #Cell dofs and displacement
        celldofs = zeros(Int, ndofs_per_cell(dh,cellid))
        celldofs!(celldofs, dh, cellid)
        ue = state.d[celldofs]
        
        #Coords of cell
        JuAFEM.cellcoords!(X, dh, cellid)
        Xᵇ = IGA.compute_bezier_points(Ce, X)

        #Shape values for evaluating stresses at center of cell
        cv_sr = deepcopy(igashell.integration_data.cell_values_sr)
        oop_values = _build_oop_basisvalue!(igashell, cellstate)
        
        set_bezier_operator!(cv_sr, Ce)
        set_oop_basefunctions!(cv_sr, oop_values)
        
        reinit!(cv_sr, Xᵇ)
        active_layer_dofs = build_active_layer_dofs(igashell, cellstate)

        @assert(nqp_outofplane == getnquadpoints(cv_sr))
        iqp = 0
        for ilay in 1:nlayers(igashell)
            active_dofs = active_layer_dofs[ilay]
            ue_layer = ue[active_dofs]
            
            for _ in 1:getnquadpoints_ooplane_per_layer(igashell)
                iqp +=1 
                σ, x_glob, x_loc = _eval_stress_center(cv_sr, igashell.layerdata.layer_materials[ilay], iqp, Xᵇ, ue_layer, active_dofs, small_deformations)

                push!(ts.local_coords, x_loc)
                push!(ts.global_coords, x_glob)
                push!(ts.stresses, σ)
            end
        end
        push!(ttstresses, ts)
    end
    
    return ttstresses
end

function _eval_stress_center(cv::IGAShellValues{dim_s,dim_p,T}, material, qp, Xᵇ, ue, active_dofs, small_deformation_theroy::Bool) where {dim_s,dim_p,T}

    g = zeros(Vec{dim_s,T}, dim_s)

    x_glob = spatial_coordinate(cv, qp, Xᵇ)
    R = cv.R[qp]
    x_loc = R' ⋅ x_glob

    #Covarient triad on deformed configuration
    for d in 1:dim_s
        g[d] = cv.G[qp][d] + function_parent_derivative(cv, qp, ue, d, active_dofs)
    end

    # Deformation gradient
    F = zero(Tensor{2,dim_s,T})
    for i in 1:dim_s
        F += g[i]⊗cv.Gᴵ[qp][i]
    end

    #Construct a fake material state
    #This is only works if it is elastic materialstate
    matstate = Five.getmaterialstate(material)

    #Eval strains
    if small_deformation_theroy
        ɛ = symmetric(F) - one(SymmetricTensor{2,dim_s})
        _̂ε = symmetric(R' ⋅ ɛ ⋅ R)
        _̂σ, ∂̂σ∂ɛ, new_matstate = Five.constitutive_driver(material, _̂ε, matstate)
    else
        U = sqrt(tdot(F))
        E = symmetric(1/2 * (F' ⋅ F - one(F)))
        S, ∂S∂E, new_matstate = Five.constitutive_driver(material, E, matstate)
        _̂σ = inv(det(F)) * U ⋅ S ⋅ U
    end
    
    if dim_s == 2
        x_glob = Vec{3}((x_glob[1], 0.0, x_glob[2]))
        x_loc = Vec{3}((x_loc[1], 0.0, x_loc[2]))
    end

    return new_matstate.σ, x_glob, x_loc
end

"""
IGAShellRecovoredStressOutput
    Output
"""
struct IGAShellRecovoredStressOutput{P<:IGAShell} <: Five.AbstractOutput
    igashell::Base.RefValue{P}
end

function IGAShellRecovoredStressOutput(; igashell::IGAShell)
    return IGAShellRecovoredStressOutput(Base.RefValue(igashell))
end

function Five.build_outputdata(output::IGAShellRecovoredStressOutput, set, ::MixedDofHandler)
    @assert(eltype(set) == Int) # Only accept cellids
    return output
end

function Five.collect_output!(output::IGAShellRecovoredStressOutput, state::StateVariables{T}, cellset, globaldata) where T
    
    #Extract some variables
    igashell = output.igashell[]
    dh = globaldata.dh
    dim_s = JuAFEM.getdim(igashell)

    #pre-allocate variables
    nnodes = JuAFEM.nnodes_per_cell(igashell)
    X = zeros(Vec{dim_s,T}, nnodes)

    cv_sr = intdata(igashell).cell_values_sr

    recovered_stresses = [RecoveredStresses{T}[] for _ in cellset]
    for (ic, cellid) in enumerate(cellset)
        local_id = findfirst((i)->i==cellid, igashell.cellset)
        stresses = igashell.stress_recovory.recovered_stresses[:, local_id]
        recovered_stresses[ic] = stresses
    end
    
    return recovered_stresses

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
    for (ic, faceidx) in enumerate(faceset)

        cellid = faceidx[1]
        local_cellid = findfirst((i)->i==cellid, igashell.cellset)
        
        _celldofs = zeros(Int, JuAFEM.ndofs_per_cell(dh, cellid))
        JuAFEM.celldofs!(_celldofs, dh, cellid)
        ue = state.d[_celldofs]

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
        reinit!(cv, Xᵇ)

        for qp in 1:getnquadpoints(cv)
            u = function_value(cv, qp, ue)
            for d in output.components
                maxu = max(maxu, abs(u[d]))
            end
        end

        #Loop through all basefunctions
        for dof in 1:getnbasefunctions(cv) ÷ dim_s
            #Check if it is not zero
            if !(basis_value(cv, 1, (dof-1)*dim_s + 1)[1] ≈ 0.0)
                for d in output.components
                    push!(alldofs, _celldofs[(dof-1)*dim_s + d])
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

