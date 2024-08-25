@inline zeroforcing(args...) = 0

"""
    regularize_forcing(forcing, field, field_name, model_field_names)

"Regularizes" or "adds information" to user-defined forcing objects that are passed to
model constructors. `regularize_forcing` is called inside `model_forcing`.

We need `regularize_forcing` because it is only until `model_forcing` is called that
the fields (and field locations) of various forcing functions are available. The `field`
can be used to infer the location at which the forcing is applied, or to add a field
dependency to a special forcing object, as for `Relxation`.
"""
regularize_forcing(forcing, args...) = forcing # fallback
# TODO: some checking that `array` is validly-sized could be done here
regularize_forcing(array::AbstractArray, args...) = Forcing(array)
regularize_forcing(::Nothing, args...) = zeroforcing

"""
    regularize_forcing(forcing::Function, field, field_name, model_field_names)

Wrap `forcing` in a `ContinuousForcing` at the location of `field`.
"""
function regularize_forcing(forcing::Function, field, args...)
    LX, LY, LZ = location(field)
    return ContinuousForcing{LX, LY, LZ}(forcing)
end

"""
    model_forcing(model_fields; forcings...)

Return a `NamedTuple` of forcing functions for each field in `model_fields`, wrapping
forcing functions in `ContinuousForcing`s and ensuring that `ContinuousForcing`s are
located correctly for each field.
"""
function model_forcing(model_fields; user_forcings...)

    model_field_names = keys(model_fields)

    regularized_forcings = Tuple(
        name in keys(forcings) ?
            regularize_forcing(forcings[name], field, name, model_field_names) : zeroforcing
        for (name, field) in pairs(model_fields)
    )

    specified_forcings = NamedTuple{model_field_names}(regularized_forcings)

    return specified_forcings
end
