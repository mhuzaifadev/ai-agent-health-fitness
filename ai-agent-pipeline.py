import asyncio
import json
import os
from typing import List
from dataclasses import dataclass
from pydantic import BaseModel, Field
from agents import Agent, Runner, function_tool, InputGuardrail, GuardrailFunctionOutput, InputGuardrailTripwireTriggered
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set model choice
model = os.getenv('LLM_MODEL_NAME', 'gpt-4.1-mini')

# --- Detailed Data Models ---
class WorkoutPlan(BaseModel):
    """Workout recommendation with detailed parameters"""
    focus_area: str = Field(description="Primary focus of the workout (e.g., 'upper body', 'cardio')")
    difficulty: str = Field(description="Difficulty level (Beginner, Intermediate, Advanced)")
    exercises: List[str] = Field(description="List of recommended exercises")
    notes: str = Field(description="Additional tips and form recommendations")

class MealPlan(BaseModel):
    """Meal plan recommendation with macronutrient targets"""
    daily_calories: int = Field(description="Recommended daily calorie intake")
    protein_grams: int = Field(description="Daily protein target in grams")
    carbs_grams: int = Field(description="Daily carbohydrate target in grams")
    fat_grams: int = Field(description="Daily fat target in grams")
    meal_suggestions: List[str] = Field(description="Meal suggestions")
    notes: str = Field(description="Dietary advice and nutritional tips")

class GoalAnalysis(BaseModel):
    """Analysis of user fitness goals to determine realism and safety"""
    is_realistic: bool = Field(description="Whether the fitness goal is realistic and healthy")
    reasoning: str = Field(description="Explanation of the analysis")

# --- Extended User Context ---
@dataclass
class UserContext:
    user_id: str
    fitness_level: str  # e.g., Beginner, Intermediate, Advanced
    fitness_goal: str   # e.g., Weight loss, Muscle gain, General fitness
    dietary_preference: str  # e.g., Vegan, Vegetarian, No restrictions
    available_equipment: List[str]
    weight_kg: float = 70.0       # Additional personal metric for calorie calculations
    height_cm: float = 170.0      # Additional personal metric for calorie calculations
    age: int = 30               # Additional personal metric for calorie calculations
    gender: str = "male"        # Additional personal metric for calorie calculations


@function_tool
def get_exercise_info(muscle_group: str) -> str:
    """Get a list of exercises for a specific muscle group along with recommendations"""
    exercise_data = {
        "chest": [
            "Push-ups: 3 sets of 10-15 reps",
            "Bench Press: 3 sets of 8-12 reps",
            "Chest Flyes: 3 sets of 12-15 reps",
            "Incline Push-ups: 3 sets of 10-15 reps"
        ],
        "back": [
            "Pull-ups: 3 sets of 6-10 reps",
            "Bent-over Rows: 3 sets of 8-12 reps",
            "Lat Pulldowns: 3 sets of 10-12 reps",
            "Superman Holds: 3 sets of 30 seconds"
        ],
        "legs": [
            "Squats: 3 sets of 10-15 reps",
            "Lunges: 3 sets of 10 per leg",
            "Calf Raises: 3 sets of 15-20 reps",
            "Glute Bridges: 3 sets of 15 reps"
        ],
        "arms": [
            "Bicep Curls: 3 sets of 10-12 reps",
            "Tricep Dips: 3 sets of 10-15 reps",
            "Hammer Curls: 3 sets of 10-12 reps",
            "Overhead Tricep Extensions: 3 sets of 10-12 reps"
        ],
        "core": [
            "Planks: 3 sets of 30-60 seconds",
            "Crunches: 3 sets of 15-20 reps",
            "Russian Twists: 3 sets of 20 total reps",
            "Mountain Climbers: 3 sets of 20 total reps"
        ]
    }
    muscle_group = muscle_group.lower()
    if muscle_group in exercise_data:
        exercises = exercise_data[muscle_group]
        return json.dumps({
            "muscle_group": muscle_group,
            "exercises": exercises,
            "recommendation": f"For {muscle_group} training, complete exercises with 60-90 seconds rest between sets."
        })
    else:
        return f"Exercise information for {muscle_group} is not available."

@function_tool
def calculate_calories(goal: str, weight_kg: float, height_cm: float, age: int, gender: str) -> str:
    """Calculate daily calorie needs and provide macronutrient breakdown based on user stats and goals"""
    # Calculate BMR using Mifflin-St Jeor Equation
    if gender.lower() in ['male', 'm']:
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
    else:
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161

    # Use a moderate activity factor
    tdee = bmr * 1.55

    # Adjust calorie target based on goal
    if goal.lower() == "weight loss":
        calorie_target = tdee - 500  # Rough deficit for weight loss
    elif goal.lower() == "muscle gain":
        calorie_target = tdee + 300  # Rough surplus for muscle gain
    else:
        calorie_target = tdee

    # Macro breakdown (simplified approach)
    if goal.lower() == "weight loss":
        protein_pct, fat_pct, carb_pct = 0.40, 0.30, 0.30
    elif goal.lower() == "muscle gain":
        protein_pct, fat_pct, carb_pct = 0.30, 0.25, 0.45
    else:
        protein_pct, fat_pct, carb_pct = 0.30, 0.30, 0.40

    protein_cal = calorie_target * protein_pct
    fat_cal = calorie_target * fat_pct
    carb_cal = calorie_target * carb_pct

    protein_grams = round(protein_cal / 4)
    fat_grams = round(fat_cal / 9)
    carb_grams = round(carb_cal / 4)

    result = {
        "goal": goal,
        "daily_calories": round(calorie_target),
        "macros": {
            "protein": protein_grams,
            "fat": fat_grams,
            "carbs": carb_grams
        }
    }
    return json.dumps(result)

# --- Guardrail for Fitness Goals ---
goal_analysis_agent = Agent(
    name="Goal Analyzer",
    instructions="""
    Analyze the user's fitness goal. Losing more than 2 pounds per week is generally unsafe.
    """,
    output_type=GoalAnalysis,
    model=model
)

async def fitness_goal_guardrail(ctx, agent, input_data):
    """Check if the user's fitness goal is realistic and safe."""
    try:
        analysis_prompt = f"The user said: {input_data}. Analyze whether this fitness goal is realistic and safe."
        result = await Runner.run(goal_analysis_agent, analysis_prompt)
        final_output = result.final_output_as(GoalAnalysis)
        return GuardrailFunctionOutput(
            output_info=final_output,
            tripwire_triggered=not final_output.is_realistic,
        )
    except Exception as e:
        return GuardrailFunctionOutput(
            output_info=GoalAnalysis(is_realistic=True, reasoning=f"Error analyzing goal: {str(e)}"),
            tripwire_triggered=False
        )

# --- Specialized Agents with Handoffs ---

# Workout Specialist: Uses the exercise information tool and user context to provide a personalized plan
workout_agent = Agent[UserContext](
    name="Workout Specialist",
    handoff_description="Creates personalized workout plans using detailed exercise info.",
    instructions="""
    You are a workout specialist. Use the user's fitness level, goal, and available equipment to design a workout plan.
    Leverage the get_exercise_info tool to retrieve specific exercise recommendations.
    Provide a clear focus area (e.g., 'upper body', 'cardio') and a difficulty level (Beginner, Intermediate, Advanced) along with form tips.
    """,
    model=model,
    tools=[get_exercise_info],
    output_type=WorkoutPlan
)

# Nutrition Specialist: Uses the calorie calculation tool to provide a meal plan
nutrition_agent = Agent[UserContext](
    name="Nutrition Specialist",
    handoff_description="Creates personalized meal plans with calorie and macro targets.",
    instructions="""
    You are a nutrition specialist. Use the user's fitness goal, dietary preferences, and personal stats to calculate daily calorie needs.
    Leverage the calculate_calories tool to compute calorie targets and macronutrient breakdown.
    Suggest practical meals that help reach the user's nutrition goals.
    """,
    model=model,
    tools=[calculate_calories],
    output_type=MealPlan
)

# Main Fitness Agent with Guardrails and Handoffs
fitness_agent = Agent[UserContext](
    name="Robust Fitness Coach",
    instructions="""
    You are a holistic fitness coach. Process general fitness queries while checking the realism of the user's fitness goals.
    When specific inquiries about workouts or nutrition arise, hand off to the Workout or Nutrition Specialists respectively.
    Use the provided user context and guardrail function to ensure safety.
    """,
    model=model,
    handoffs=[workout_agent, nutrition_agent],
    input_guardrails=[InputGuardrail(guardrail_function=fitness_goal_guardrail)]
)

# --- Demo Function to Showcase the New Agent ---
async def demo():
    # Create a user context with extended personal details
    user_context = UserContext(
        user_id="user123",
        fitness_level="beginner",
        fitness_goal="I want to lose 20 pounds in 2 weeks",  # This is unrealistic and should trigger the guardrail
        dietary_preference="no restrictions",
        available_equipment=["dumbbells", "resistance bands"],
        weight_kg=80.0,
        height_cm=175.0,
        age=28,
        gender="male"
    )
    
    queries = [
        "I want to start working out to lose weight. What exercises should I do?",
        "How should I eat to build muscle and support my training?",
        "I want to lose 20 pounds in 2 weeks"
    ]
    
    for query in queries:
        print("\n" + "="*50)
        print(f"QUERY: {query}")
        print("="*50)
        try:
            result = await Runner.run(fitness_agent, query, context=user_context)
            
            # Identify which agent provided the answer
            if isinstance(result.final_output, WorkoutPlan):
                print("\n[👟 WORKOUT SPECIALIST]")
            elif isinstance(result.final_output, MealPlan):
                print("\n[🍎 NUTRITION SPECIALIST]")
            else:
                print("\n[🏋️ GENERAL FITNESS COACH]")
            
            print("RESPONSE:")
            print(result.final_output)
            
        except InputGuardrailTripwireTriggered as e:
            print("\n[⚠️ GUARDRAIL TRIGGERED]")
            if hasattr(e, 'guardrail_output') and hasattr(e.guardrail_output, 'reasoning'):
                print(f"Reason: {e.guardrail_output.reasoning}")
            else:
                print("An unrealistic or unsafe fitness goal was detected.")
        except Exception as e:
            print(f"\n[❌ ERROR]: {str(e)}")

if __name__ == "__main__":
    asyncio.run(demo())
