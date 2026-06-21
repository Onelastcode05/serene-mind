# serene-mind

# Phase-1
> ML model predicting stress score based on various factors concerning the user.
> The model also extracts the top features affecting the stress score.
> Problems Encountered : The dynamic factors absent from the schema are creating a problem. The stress score calculation is affected and inaccurate for these.

# Phase-2
> A lang graph implemented using Ollama Phi-4 LLM.
> A proper structured agent returning insights in JSON format taking the stress score and top 5 features from the phase-1 model.
> The lang graph contains the planner and analyzer node.
> The planner node basically assigns a safety level to the user, and sets the agent accordingly.
> The analyzer node formulates the final outcomes we require.
> Problems Encountered : The ollama model runs locally. When we will try to move it to cloud, we will have to setup ollama on a seperate server and set endpoints through APIs. A cloud guide has been provided for that.

 ### Review and testing in progress...
