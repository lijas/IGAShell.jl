export IGAShellExternalForce
#
#
#
struct IGAShellExternalForce{P<:IGAShell} <: Five.AbstractExternalForce
    faceset::GeometryObjectVectors
    traction::Function
    igashell::Base.RefValue{P}
end

function IGAShellExternalForce(; 
    set, 
    func::Function, 
    igashell::IGAShell)

    return IGAShellExternalForce{typeof(igashell)}(collect(set), func, Base.RefValue(igashell))
end

function Five.init_external_force!(a::IGAShellExternalForce, ::JuAFEM.AbstractDofHandler)
    return a
end

function Five.apply_external_force!(ef::IGAShellExternalForce{P}, state::StateVariables{T}, globaldata) where {P,T}
    
    #Igashell extract
    dh = globaldata.dh
    igashell = ef.igashell[]
    dim_s = JuAFEM.getdim(igashell)

    #Coords
    ncoords = JuAFEM.nnodes_per_cell(igashell)
    X = zeros(Vec{dim_s,T}, ncoords)
    Xᵇ = zeros(Vec{dim_s,T}, ncoords)

        
    ndofs_layer = ndofs_per_layer(igashell)
    fe = zeros(T, ndofs_layer)
    ke = zeros(T, ndofs_layer, ndofs_layer)

    #Traction force
    traction = ef.traction
    A = 0.0
    for faceidx in ef.faceset
        cellid, faceid = faceidx
        ic = findfirst((x)->x==cellid, igashell.cellset)

        ilay = (faceidx == 1) ? 1 : nlayers(igashell)

        #
        cellstate = getcellstate(adapdata(igashell), ic)
        active_layerdofs = build_active_layer_dofs(igashell, cellstate)
        ndofs = ndofs_per_cell(dh, cellid)
        celldofs = zeros(Int, ndofs)

        #
        Ce = get_extraction_operator(intdata(igashell), ic)
        
        #Get coords and dofs of cell
        JuAFEM.cellcoords!(X, dh, cellid)
        JuAFEM.celldofs!(celldofs, dh, cellid)

        Xᵇ .= IGA.compute_bezier_points(Ce, X)

        #
        @timeit "build" cv = build_facevalue!(igashell, faceidx)

        IGA.set_bezier_operator!(cv, Ce)
        reinit_midsurface!(cv, Xᵇ)

        for ilay in 1:nlayers(igashell)
            fill!(fe, 0.0)

            layerdofs = celldofs[active_layerdofs[ilay]]

            reinit_layer!(cv, ilay)

            @timeit "integrate" A += _compute_igashell_external_traction_force!(cv, Xᵇ, traction, faceidx, state.t, fe, getwidth(layerdata(igashell)))
            state.system_arrays.fᵉ[layerdofs] += fe
        end
    end

end

function _compute_igashell_external_traction_force!(fv::IGAShellValues, Xᵇ, traction::Function, faceidx, time::Float64, fe::AbstractVector, width)

    A = 0.0
    for q_point in 1:getnquadpoints_per_layer(fv)
        dΓ = getdetJdA(fv, q_point, faceidx) * width
        X = spatial_coordinate(fv, q_point, Xᵇ)

        t = traction(X,time)
        A += dΓ
        for i in 1:getnbasefunctions_per_layer(fv)
            δui = basis_value(fv, q_point, i)
            fe[i] += (δui ⋅ t) * dΓ
        end
    end

    return A
end