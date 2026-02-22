async def agent_execution_loop(task):
    context = prepare_context(task, instructions, memory, history)

    while not done:
        response = await model_client.create(context)

        if response.has_tool_calls:
            for tool_call in response.tool_calls:
                result = await execute_tool(tool_call)
                context.append(result)
        else:
            done = True
    update_memory(context) # optional 
    return response