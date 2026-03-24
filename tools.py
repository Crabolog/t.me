tools = [
    {
        "type": "function",
        "name": "search_and_extract",
        "description": "Searches for information using SerpAPI (Google Search API)",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string",
                },
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "reboot_pi",
        "description": "Reboots the Raspberry Pi asynchronously.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "git_pull",
        "description": "Asynchronously pulls the latest changes from the 'master' branch.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "update_system",
        "description": "Оновлює системний промпт, який задає поведінку бота.",
        "parameters": {
            "type": "object",
            "properties": {
                "new_prompt": {
                    "type": "string",
                    "description": "Новий system prompt для бота"
                }
            },
            "required": ["new_prompt"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "generate_image",
        "description": "Генерує зображення за описом",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string"
                }
            },
            "required": ["prompt"],
            "additionalProperties": False
        },
        "strict": True
    }
]