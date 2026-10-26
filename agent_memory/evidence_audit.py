"""Evaluation-only observable evidence stages. Never imported by retrieval/reader."""

def selection_trace(gold,retrieved,packed,cited,*,gold_images=(),packed_images=(),sent_images=()):
    """ID/occurrence accounting only, not causal attribution or model attention.

    Image keys must identify occurrences, not just blob hashes. Gold can only be
    supplied by a separate post-answer evaluator. Empty gold is not zero recall.
    """
    required=set(gold);got=set(retrieved);context=set(packed);quotes=set(cited)
    if not context<=got or not quotes<=context:raise ValueError('observable stages do not align')
    needed_images=set(gold_images);available_images=set(packed_images);selected_images=set(sent_images)
    if not selected_images<=available_images:raise ValueError('sent image was not available in packed evidence')
    def fraction(values,required):return len(values&required)/len(required) if required else None
    return dict(gold_count=len(required),retrieved_gold_count=len(required&got),packed_gold_count=len(required&context),
        cited_gold_count=len(required&quotes),retrieval_fraction=fraction(got,required),context_fraction=fraction(context,required),
        cited_fraction=fraction(quotes,required),gold_lost_during_packing=sorted((required&got)-context),
        packed_gold_not_cited=sorted((required&context)-quotes),text_prefix_dropped=len(got-context),
        gold_image_occurrences=len(needed_images),gold_images_available=len(needed_images&available_images),
        gold_images_sent=len(needed_images&selected_images),available_gold_images_not_sent=len((needed_images&available_images)-selected_images),
        cited_recall_is_not_internal_usage=True,annotated_round_images_not_necessarily_all_required=True)
