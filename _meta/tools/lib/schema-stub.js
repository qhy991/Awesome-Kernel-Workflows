'use strict'
// Pure schemaStub(schema) — no I/O, no globals.
// Generates a minimal valid stub value from a JSON Schema object.
// LOAD-BEARING: both 'number' AND 'integer' map to 0.
//   AccelOpt L406/L442 call .toFixed(3) on latency stubs; anything other than a number crashes.

/**
 * @param {object|null|undefined} schema — a JSON Schema (sub-)object
 * @returns a minimal valid value for that schema
 */
function schemaStub(schema) {
  if (!schema || typeof schema !== 'object') return {}

  // enum: return first element
  if (Array.isArray(schema.enum) && schema.enum.length > 0) {
    return schema.enum[0]
  }

  const type = schema.type

  switch (type) {
    case 'string':  return ''
    case 'number':  return 0   // LOAD-BEARING: .toFixed() must not throw
    case 'integer': return 0   // integer→0; retrofit schemas may introduce integer type
    case 'boolean': return false
    case 'array':   return []

    case 'object': {
      const result = {}
      const props = schema.properties || {}

      // Fill all declared properties
      for (const [key, subSchema] of Object.entries(props)) {
        result[key] = schemaStub(subSchema)
      }

      // Fill any required keys not already covered by properties
      if (Array.isArray(schema.required)) {
        for (const key of schema.required) {
          if (!(key in result)) {
            result[key] = schemaStub(null)
          }
        }
      }

      // additionalProperties:true only (no properties declared) => {}
      return result
    }

    default:
      // No type declared — return {}
      return {}
  }
}

module.exports = schemaStub
