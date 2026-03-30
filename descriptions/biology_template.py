"""
Biology Class Descriptions Template

This file provides a template for creating class descriptions for biological
species classification. Class descriptions help the model understand visual
characteristics of each species and improve classification accuracy.

Usage:
    1. Copy this file
    2. Update the 'classes' list with your species names
    3. Write detailed descriptions for each species
    4. Reference in config: classification.desc_text_path: "path/to/descriptions.py"

Guidelines for writing descriptions:
    - Focus on diagnostic features (what makes this species unique)
    - Include visual characteristics: colors, patterns, size, shape
    - Mention distinctive markings or field marks
    - Include habitat clues if visible in images
    - Keep descriptions 2-4 sentences for optimal performance
"""

# List of class names (must match your support set folder names)
classes = [
    "example_species_a",
    "example_species_b",
    "example_species_c",
]

# Detailed descriptions for each species
# These descriptions are used to generate text features for classification
descriptions = {
    "example_species_a": (
        "Description of example species A. "
        "Include key visual characteristics like coloration, patterns, size, and distinctive marks. "
        "Mention any unique features that help identify this species."
    ),
    "example_species_b": (
        "Description of example species B. "
        "Focus on diagnostic features that distinguish it from similar species. "
        "Include details about body shape, markings, and coloration."
    ),
    "example_species_c": (
        "Description of example species C. "
        "Describe visual field marks and characteristics. "
        "Mention any seasonal variations if applicable."
    ),
}


# =============================================================================
# EXAMPLE: Butterfly Species
# =============================================================================

BUTTERFLY_EXAMPLE = {
    "classes": [
        "monarch",
        "viceroy",
        "swallowtail_tiger",
        "painted_lady",
    ],
    "descriptions": {
        "monarch": (
            "Large orange butterfly with black veins creating a stained-glass pattern. "
            "Wings have white spots along the black borders. "
            "Orange color is bright and uniform. "
            "Body is black with white spots."
        ),
        "viceroy": (
            "Orange butterfly similar to monarch but with black line across hindwing. "
            "Smaller than monarch with more pointed wings. "
            "Orange color is deeper and less bright than monarch. "
            "Key feature is the curved black line on lower wing."
        ),
        "swallowtail_tiger": (
            "Large yellow butterfly with black tiger stripes. "
            "Distinctive tail-like extensions on hindwings. "
            "Wings have blue and orange spots near the tails. "
            "One of the largest butterflies in North America."
        ),
        "painted_lady": (
            "Orange-brown butterfly with black and white markings. "
            "Wings have distinctive four small eyespots on hindwing. "
            "Coloration is more muted and patchy than monarch. "
            "Wings have pinkish-orange patches with black spots."
        ),
    },
}


# =============================================================================
# EXAMPLE: Bird Species
# =============================================================================

BIRD_EXAMPLE = {
    "classes": [
        "american_goldfinch",
        "american_robin",
        "blue_jay",
        "northern_cardinal",
    ],
    "descriptions": {
        "american_goldfinch": (
            "Small bright yellow finch with black cap and wings. "
            "White wing bars and black wing feathers. "
            "Conical orange bill perfect for seed-eating. "
            "Male breeding plumage is vibrant yellow. "
            "Undulating flight pattern."
        ),
        "american_robin": (
            "Medium-sized thrush with reddish-orange breast. "
            "Dark head, back, and wings with white eye rings. "
            "Yellow bill and grayish underparts. "
            "Often seen running and stopping on lawns. "
            "Distinctive cheer-up cheer-up song."
        ),
        "blue_jay": (
            "Large crested songbird with bright blue upperparts. "
            "White face and underparts with black necklace. "
            "Distinctive blue crest on head. "
            "Black bars on wings and tail. "
            "Loud jay-jay call."
        ),
        "northern_cardinal": (
            "Medium-sized songbird with prominent crest on head. "
            "Male is bright red all over with black face mask. "
            "Female is tan-brown with red wings, tail, and crest. "
            "Both sexes have thick orange-red conical bill. "
            "Often seen at bird feeders."
        ),
    },
}


# =============================================================================
# EXAMPLE: Mammal Species (Camera Trap)
# =============================================================================

MAMMAL_EXAMPLE = {
    "classes": [
        "white_tailed_deer",
        "black_bear",
        "coyote",
        "red_fox",
    ],
    "descriptions": {
        "white_tailed_deer": (
            "Medium-sized deer with reddish-brown coat in summer. "
            "Distinctive white underside of tail visible when raised. "
            "Large ears and white throat patch. "
            "Males have antlers (shed in winter). "
            "Often seen in open areas and forest edges."
        ),
        "black_bear": (
            "Large stocky mammal with black fur (may have brown muzzle). "
            "Prominent shoulder hump and rounded ears. "
            "Short tail and plantigrade feet. "
            "Small eyes and long snout. "
            "Walks with distinctive flat-footed gait."
        ),
        "coyote": (
            "Medium-sized canine with grayish-brown coat. "
            "Pointed ears and narrow muzzle. "
            "Bushy tail often held down (not curled). "
            "Long legs relative to body size. "
            "Smaller and more slender than wolf."
        ),
        "red_fox": (
            "Small to medium canine with distinctive red-orange coat. "
            "White underparts and black legs. "
            "Bushy tail with white tip. "
            "Pointed ears and white cheek patches. "
            "Black markings on ears and muzzle."
        ),
    },
}


# =============================================================================
# EXAMPLE: Insect Species
# =============================================================================

INSECT_EXAMPLE = {
    "classes": [
        "honey_bee",
        "bumble_bee",
        "carpenter_bee",
        "paper_wasp",
    ],
    "descriptions": {
        "honey_bee": (
            "Medium-sized bee with golden-brown and black stripes. "
            "Relatively slim abdomen with alternating bands. "
            "Hairy thorax and translucent wings. "
            "Pollen baskets visible on hind legs when carrying. "
            "Smaller and more slender than bumble bee."
        ),
        "bumble_bee": (
            "Large fuzzy bee with black and yellow bands. "
            "Very hairy body covering most of the abdomen. "
            "Round robust body shape. "
            "Loud buzzing flight. "
            "Larger than honey bees with more vivid coloration."
        ),
        "carpenter_bee": (
            "Large bee resembling bumble bee but with shiny black abdomen. "
            "Yellow thorax with black hairs. "
            "Less fuzzy than bumble bee. "
            "Distinctive shiny hairless abdomen. "
            "Often seen hovering near wooden structures."
        ),
        "paper_wasp": (
            "Slender wasp with long legs hanging in flight. "
            "Narrow waist connecting thorax and abdomen. "
            "Yellow and brown markings. "
            "Smooth less hairy body than bees. "
            "Distinctive umbrella-shaped nest if visible."
        ),
    },
}


# =============================================================================
# TEMPLATE FOR YOUR PROJECT
# =============================================================================

# Uncomment and modify for your specific use case:

# classes = [
#     "your_species_1",
#     "your_species_2",
#     "your_species_3",
# ]
#
# descriptions = {
#     "your_species_1": (
#         "Description focusing on diagnostic features. "
#         "Colors, patterns, and distinctive markings. "
#         "Size and shape characteristics."
#     ),
#     "your_species_2": (
#         "Description focusing on diagnostic features. "
#         "What distinguishes it from similar species. "
#         "Key field marks for identification."
#     ),
#     "your_species_3": (
#         "Description focusing on diagnostic features. "
#         "Visual characteristics and unique traits. "
#         "Habitat and behavioral clues if visible."
#     ),
# }
