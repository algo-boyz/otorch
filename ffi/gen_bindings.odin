package main

import "core:fmt"
import "core:os"
import "core:strings"

INPUT_FILE :: "torch_api_generated.h"

main :: proc() {
    data, ok := os.read_entire_file(INPUT_FILE)
    if !ok {
        fmt.eprintln("Error: Could not read", INPUT_FILE)
        return
    }
    defer delete(data)

    content := string(data)
    
    fmt.println("// AUTOMATICALLY GENERATED BINDINGS")
    fmt.println("package otorch")
    fmt.println("")
    fmt.println("foreign import lib \"libtorch_wrapper.dylib\"")
    fmt.println("")
    fmt.println("foreign lib {")

    it := content
    for line in strings.split_lines_iterator(&it) {
        trimmed := strings.trim_space(line)
        
        // We only care about lines starting with "void atg_"
        if strings.has_prefix(trimmed, "void atg_") {
            parse_and_print_func(trimmed)
        }
    }

    fmt.println("}")
}

parse_and_print_func :: proc(line: string) {
    // Expected format: void atg_name(args...);
    
    // Remove "void " prefix (5 chars)
    without_void := line[5:] 
    
    paren_start := strings.index(without_void, "(")
    if paren_start == -1 do return

    func_name := without_void[:paren_start]

    // Extract args
    paren_end := strings.index(without_void, ")")
    if paren_end == -1 do return
    
    args_content := without_void[paren_start+1 : paren_end]
    
    // Split by comma
    raw_args := strings.split(args_content, ",")
    defer delete(raw_args)

    fmt.printf("\t%s :: proc(", func_name)

    for arg_str, i in raw_args {
        clean_arg := strings.trim_space(arg_str)
        if len(clean_arg) == 0 do continue

        if i > 0 do fmt.print(", ")

        c_type, name := split_type_and_name(clean_arg)
        
        if i == 0 && name == "" && strings.contains(c_type, "*") {
            name = "out"
        } else if name == "" {
            // Fallback for unnamed args
            name = fmt.tprintf("arg%d", i) 
        }
        odin_type := map_c_type_to_odin(c_type)
        fmt.printf("%s: %s", name, odin_type)
    }
    fmt.print(") ---\n")
}

// Moves '*' from the variable name to the type side.
// eg "int64_t" "*out" -> "int64_t *" "out"
split_type_and_name :: proc(arg: string) -> (type: string, name: string) {
    last_space := strings.last_index(arg, " ")
    
    if last_space == -1 {
        // e.g. "tensor*" or "void"
        return arg, ""
    }

    t_part := strings.trim_space(arg[:last_space])
    n_part := strings.trim_space(arg[last_space+1:])

    // If name starts with pointers like "*ptr", move '*' to type
    for strings.has_prefix(n_part, "*") {
        n_part = n_part[1:]
        t_part = fmt.tprintf("%s *", t_part)
    }
    
    return t_part, n_part
}

map_c_type_to_odin :: proc(c_type: string) -> string {
    // Strip "const "
    t := c_type
    if strings.has_prefix(t, "const ") {
        t = t[6:]
    }
    // Normalize spacing around pointers
    // int64_t * -> int64_t*
    t = strings.join(strings.split(t, " "), "") 

    switch t {
    case "tensor*":      return "^Tensor"
    case "tensor":       return "Tensor"
    case "tensor*[]":    return "[^]Tensor" // Array of Tensor pointers
    
    case "scalar":       return "Scalar"
    case "scalar*":      return "^Scalar"

    case "int64_t":      return "i64"
    case "int64_t*":     return "[^]i64"   // Usually an array (eg sizes)
    case "double":       return "f64"
    case "double*":      return "[^]f64"
    case "bool":         return "bool"
    case "int":          return "c.int"
    case "char*":        return "cstring"
    case "void*":        return "rawptr"
    }
    // Default fallback
    return "rawptr"
}